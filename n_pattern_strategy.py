"""
N型走势量化策略（日内回测）
============================

策略核心逻辑：
  A(起涨低点) → B(第一波高点) → C(回调低点，不破A) → 突破B高点 → 开仓做多
  止损设在A点低点，收盘强制平仓，每日独立，不持仓过夜。

数据格式：parquet 分钟线，字段包含 Datetime / Open / High / Low / Close / Volume 等

核心约束：不使用未来数据 —— 所有判断仅基于当前K线及之前的历史数据
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 策略参数 — 可调参数全部集中在这里
# ============================================================
CONFIG = {
    # 枢轴点检测：一根K线要成为"枢轴点"，需要比左右各 N 根K线的极值更高/更低
    # 并且要等右侧 N 根K线走完才能确认，确保不用未来数据
    'pivot_lookback': 6,

    # 第一段上涨(A→B)的最小幅度，小于此幅度的"上涨"视为噪音，不构成N型第一笔
    'min_leg_pct': 0.003,       # 0.3%

    # 回调(B→C)幅度占第一段(A→B)的比例上限
    # 例如 0.8 表示回调最多吃掉第一段涨幅的80%，必须保留至少20%
    'max_pullback_pct': 0.8,

    # 回调(B→C)的最小幅度，必须实质性回调才构成N型
    # 如果B之后几乎没有下跌就直接突破，说明不是N型，可能只是单边拉升
    'min_pullback_pct': 0.001,  # 0.1%

    # 趋势过滤：日线收盘价 > N日均线时才允许做多，0=关闭过滤
    # 原理：只在多头趋势中做多，避开单边下跌市场
    'trend_ma_period': 0,

    # 回测起始日期（可调整）
    'start_date': '2022-01-01',

    # 数据目录与输出文件
    'data_dir': 'long',
    'output_file': 'n_pattern_results.csv',
}


# ============================================================
# 第1步：枢轴点检测
# ============================================================
def find_pivots(df, lookback):
    """
    检测K线图中的局部高点和局部低点（枢轴点）。

    原理：
      对于一根K线 i，如果它的最高价 > 左边 lookback 根K线的最高价
                        且 ≥ 右边 lookback 根K线的最高价
      则 i 被标记为「枢轴高点」。

      同理，如果它的最低价 < 左边 lookback 根K线的最低价
                        且 ≤ 右边 lookback 根K线的最低价
      则 i 被标记为「枢轴低点」。

    无未来数据保证：
      一根K线要在它发生 lookback 根K线之后才会被确认。
      例如 lookback=6，那么第10根K线是否为枢轴点，要等到第16根K线走完才能断定。
      在此之前，我们不使用这个枢轴点。
      代码中通过 `for i in range(lookback, n - lookback)` 来保证不越界读取。

    参数：
      df:       包含 High, Low 列的 DataFrame
      lookback: 左右确认窗口大小（K线数量）

    返回：
      pivot_highs: bool 数组，True 表示该索引是枢轴高点
      pivot_lows:  bool 数组，True 表示该索引是枢轴低点
    """
    n = len(df)
    pivot_highs = np.zeros(n, dtype=bool)
    pivot_lows = np.zeros(n, dtype=bool)

    highs = df['High'].values
    lows = df['Low'].values

    for i in range(lookback, n - lookback):
        # --- 检测枢轴高点 ---
        # 左侧最大值：i 之前 lookback 根K线的最高价（不含 i 本身）
        left_max = highs[i - lookback:i].max()
        # 右侧最大值：i 之后 lookback 根K线的最高价（不含 i 本身）
        right_max = highs[i + 1:i + 1 + lookback].max()
        # i 的高点必须严格大于左侧，且 ≥ 右侧（允许平顶但不允许被超越）
        if highs[i] > left_max and highs[i] >= right_max:
            pivot_highs[i] = True

        # --- 检测枢轴低点 ---
        # 同理，i 的低点必须严格小于左侧，且 ≤ 右侧
        left_min = lows[i - lookback:i].min()
        right_min = lows[i + 1:i + 1 + lookback].min()
        if lows[i] < left_min and lows[i] <= right_min:
            pivot_lows[i] = True

    return pivot_highs, pivot_lows


# ============================================================
# 第2步：N型形态识别 + 交易信号生成
# ============================================================
def find_n_pattern_signals(df, config):
    """
    在单日分钟K线数据中扫描 N 型走势，生成做多信号。

    N型形态的时间顺序（从左到右）：
      A（起涨低点）→ B（第一波高点）→ C（回调低点，高于A）
      → 当前K线突破B ← 触发开仓信号

    关键约束：
      1. C 的最低点必须 > A 的最低点（回调不创新低）
      2. B 的最高点必须 > A 的最低点（第一段是上涨）
      3. C 的最低点必须 < B 的最高点（确实发生了回调）
      4. 当前K线的最高价突破 B，触发入场
      5. 所有枢轴点都在确认后才使用（无未来数据）

    参数：
      df:     当天分钟K线 DataFrame（必须已按时间排序）
      config: 策略参数字典

    返回：
      signals: 信号列表，每个元素为 dict：
        {entry_time, entry_price, exit_time, exit_price, exit_reason, pnl, pnl_pct}
    """
    lookback = config['pivot_lookback']
    min_leg = config['min_leg_pct']
    max_pullback = config['max_pullback_pct']

    n = len(df)
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    times = df['Datetime'].values
    opens = df['Open'].values

    # 先一次性对整个 day_df 做枢轴点检测
    pivot_highs, pivot_lows = find_pivots(df, lookback)

    # 已确认的枢轴点列表（按时间顺序追加）
    # 每个元素: (K线索引, 类型 'H'=高点 或 'L'=低点)
    confirmed_pivots = []

    signals = []

    # 持仓状态
    in_trade = False
    entry_price = None      # 开仓价格
    entry_time = None       # 开仓时间
    stop_loss = None        # 止损价格（A点最低价）

    # 最小起始索引：需要足够的历史K线才能找到枢轴点
    min_start_idx = lookback * 4

    for i in range(min_start_idx, n):
        current_date = (times[i].date() if hasattr(times[i], 'date')
                        else pd.Timestamp(times[i]).date())

        # ========================================
        # 状态：已在持仓中
        # ========================================
        if in_trade:
            exit_price = None
            exit_reason = 'close'

            # --- 止损检查：当前K线最低价是否跌破A点 ---
            # A点是N型第一段起涨低点，跌破意味着形态彻底失败
            if lows[i] < stop_loss:
                exit_price = stop_loss   # 按止损价出场
                exit_reason = 'stop'

            # --- 收盘出场检查：是否是当天最后一根K线 ---
            is_last = False
            if i == n - 1:
                is_last = True  # 数据末尾，必定是当天最后一根
            else:
                next_date = (times[i + 1].date() if hasattr(times[i + 1], 'date')
                             else pd.Timestamp(times[i + 1]).date())
                if next_date != current_date:
                    is_last = True  # 下一根K线换日了

            # 如果先触发了止损就不再按收盘处理
            if exit_reason != 'stop' and is_last:
                exit_price = closes[i]
                exit_reason = 'close'

            # 满足出场条件，记录这笔交易
            if exit_price is not None:
                signals.append({
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': times[i],
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'pnl': exit_price - entry_price,
                    'pnl_pct': (exit_price - entry_price) / entry_price,
                })
                in_trade = False
            continue  # 持仓期间不做新的入场判断

        # ========================================
        # 状态：空仓，寻找入场信号
        # ========================================

        # --- 确认刚刚成熟的枢轴点 ---
        # 当前走到第 i 根K线，那么第 i-lookback 根K线的左右确认窗口已完整
        # 此时可以判断它是否为枢轴点
        confirm_idx = i - lookback
        if confirm_idx >= lookback and confirm_idx < n - lookback:
            if pivot_highs[confirm_idx]:
                confirmed_pivots.append((confirm_idx, 'H'))
            if pivot_lows[confirm_idx]:
                confirmed_pivots.append((confirm_idx, 'L'))

        # 只保留最近的枢轴点，太久远的与当前走势无关
        if len(confirmed_pivots) > 10:
            confirmed_pivots = confirmed_pivots[-10:]

        # 至少需要3个枢轴点才能构成 L→H→L 序列
        if len(confirmed_pivots) < 3:
            continue

        # --- 从已确认枢轴点中倒序查找 N 型形态 ---
        # 我们需要按时间顺序找到：L(A) → H(B) → L(C)
        # 其中 C 是最近的枢轴点（最后一个L），B在C之前（H），A在B之前（L）
        pivots = confirmed_pivots.copy()

        c_idx = None   # C点索引：回调低点（最靠近当前的枢轴低点）
        b_idx = None   # B点索引：第一波高点
        a_idx = None   # A点索引：起涨低点

        for j in range(len(pivots) - 1, -1, -1):
            idx, ptype = pivots[j]
            if ptype == 'L' and c_idx is None:
                c_idx = idx                # 找到第一个L → 作为C
            elif ptype == 'H' and c_idx is not None and b_idx is None and idx < c_idx:
                b_idx = idx                # C之前找到第一个H → 作为B
            elif ptype == 'L' and b_idx is not None and a_idx is None and idx < b_idx:
                a_idx = idx                # B之前找到第一个L → 作为A
                break                      # 找到完整序列，停止

        if a_idx is None or b_idx is None or c_idx is None:
            continue  # 序列不完整，跳过

        # ========================================
        # 验证 N 型的三个条件
        # ========================================

        a_low = lows[a_idx]     # A 点最低价（起涨点）
        b_high = highs[b_idx]   # B 点最高价（第一波顶）
        c_low = lows[c_idx]     # C 点最低价（回调底）

        # 条件1: B 必须高于 A（第一段确实是上涨）
        if b_high <= a_low:
            continue

        # 条件2: A→B 的涨幅必须达到最小阈值
        first_leg = (b_high - a_low) / a_low
        if first_leg < min_leg:
            continue

        # 条件3: C 必须高于 A（回调不破起涨点，不创新低）
        if c_low <= a_low:
            continue

        # 条件4: 回调幅度不能太大（B→C 不能吃掉 A→B 的太多）
        # pullback_ratio = (B高 - C低) / (B高 - A低)，即回调占第一段涨幅的比例
        pullback_ratio = (b_high - c_low) / (b_high - a_low)
        if pullback_ratio > max_pullback:
            continue

        # 条件5: 回调幅度必须足够（B→C 的跌幅 > 最小阈值）
        # 这是为了确保确实发生了"回调"而非横盘整理
        pullback_pct = (b_high - c_low) / b_high
        if pullback_pct < config['min_pullback_pct']:
            continue

        # 条件6: 当前K线突破 B 点高点（核心做多信号）
        if highs[i] <= b_high:
            continue

        # ========================================
        # 所有条件满足 → 触发做多信号！
        # ========================================
        entry_time = times[i]
        # 开仓价取 B点高点+一跳 与 当前K线开盘价 的较大值（保守估计）
        entry_price = max(b_high + 0.01, opens[i])
        # 止损价 = A点最低价（起涨点，跌破即形态失败）
        stop_loss = a_low
        in_trade = True

    return signals


# ============================================================
# 第3步：逐日回测
# ============================================================
def backtest_symbol(filepath, config):
    """
    对一个期货品种的完整分钟数据进行回测。

    流程：
      1. 读取 parquet 文件，按时间排序
      2. 按交易日分组（每一天是独立的）
      3. 对每天的数据运行 N 型识别
      4. 汇总所有交易信号

    趋势过滤（可选）：
      先构建日线数据，计算每日收盘价与 N 日均线的关系。
      只有前一日收盘 > 均线的日子才允许做多（只在多头趋势中交易）。

    参数：
      filepath: parquet 文件路径
      config:   策略参数

    返回：
      all_signals: 所有交易信号的列表
    """
    # 读数据，确保时间顺序
    df = pd.read_parquet(filepath)
    df = df.sort_values('Datetime').reset_index(drop=True)

    df['Date'] = df['Datetime'].dt.date
    dates = sorted(df['Date'].unique())

    # ---- 构建日线并计算趋势均线 ----
    # 日线取每日最后一根K线的收盘价作为当日收盘
    daily = df.groupby('Date').agg(
        Close=('Close', 'last'),
    ).reset_index()

    daily_close = daily['Close'].values
    daily_date = daily['Date'].values
    ma_period = config['trend_ma_period']

    # 如果启用了趋势过滤，预先计算哪些日期满足"前日收盘 > MA"条件
    # 关键：计算某日的 MA 时，只用该日之前的日线数据，不包含当日
    allowed_dates = set()
    if ma_period > 0:
        for idx in range(len(daily)):
            d = daily_date[idx]
            if d < pd.Timestamp(config['start_date']).date():
                continue
            # 至少需要 ma_period 天历史才能计算 MA
            if idx >= ma_period:
                # MA 用 idx 之前（不含当日）的 ma_period 天收盘价
                ma = daily_close[idx - ma_period:idx].mean()
                # 前一交易日收盘价 > MA → 当前日允许多头交易
                if daily_close[idx - 1] > ma:
                    allowed_dates.add(d)

    all_signals = []
    skipped = 0

    for date in dates:
        # 过滤回测起始日期之前的数据
        if date < pd.Timestamp(config['start_date']).date():
            continue

        # 趋势过滤：启用了MA但当前日期不在允许列表中 → 跳过
        if ma_period > 0 and date not in allowed_dates:
            skipped += 1
            continue

        # 提取当天数据
        day_df = df[df['Date'] == date].reset_index(drop=True)

        # 当天K线太少（不足够形成枢轴点），跳过
        if len(day_df) < config['pivot_lookback'] * 4:
            skipped += 1
            continue

        # 对该日运行 N 型识别
        signals = find_n_pattern_signals(day_df, config)

        # 给每个信号打上品种和日期的标签
        for s in signals:
            s['symbol'] = Path(filepath).stem
            s['date'] = date
        all_signals.extend(signals)

    return all_signals


# ============================================================
# 第4步：主程序入口
# ============================================================
def main():
    """主流程：读取数据 → 回测 → 输出统计结果"""
    data_dir = Path(CONFIG['data_dir'])
    files = sorted(data_dir.glob('*.parquet'))

    # ---- 选择回测品种 ----
    # 默认只跑 IC9999（中证500股指期货）
    target = 'IC9999.CCFX'
    files = [f for f in files if f.stem == target]

    if not files:
        print(f"未找到品种 {target}")
        return

    # ---- 打印参数 ----
    trend_str = (f"趋势均线={CONFIG['trend_ma_period']}日MA"
                 if CONFIG['trend_ma_period'] > 0 else "趋势过滤=关闭")
    print(f"回测品种: {target}")
    print(f"参数: 枢轴窗口={CONFIG['pivot_lookback']}, "
          f"最小涨幅={CONFIG['min_leg_pct']*100}%, "
          f"最大回调={CONFIG['max_pullback_pct']*100}%, "
          f"最小回调={CONFIG['min_pullback_pct']*100}%, "
          f"{trend_str}\n")

    # ---- 运行回测 ----
    all_results = []
    for f in files:
        symbol = f.stem
        try:
            signals = backtest_symbol(str(f), CONFIG)
            if signals:
                all_results.extend(signals)
                total_pnl = sum(s['pnl'] for s in signals)
                wins = sum(1 for s in signals if s['pnl'] > 0)
                print(f"{symbol:<20} 信号 {len(signals):>4} 个  "
                      f"胜率 {wins/len(signals)*100:5.1f}%  "
                      f"总盈亏 {total_pnl:>10.2f}")
            else:
                print(f"{symbol:<20} 无信号")
        except Exception as e:
            print(f"{symbol:<20} 出错: {e}")

    if not all_results:
        print("\n没有找到任何信号。尝试放宽参数…")
        return

    # ---- 汇总统计 ----
    results_df = pd.DataFrame(all_results)
    total_trades = len(results_df)
    wins = (results_df['pnl'] > 0).sum()
    losses = (results_df['pnl'] <= 0).sum()
    total_pnl = results_df['pnl'].sum()
    avg_pnl = results_df['pnl'].mean()
    avg_pnl_pct = results_df['pnl_pct'].mean() * 100
    max_win = results_df['pnl'].max()
    max_loss = results_df['pnl'].min()

    # 出场方式统计
    if 'exit_reason' in results_df.columns:
        stop_trades = (results_df['exit_reason'] == 'stop').sum()
        close_trades = (results_df['exit_reason'] == 'close').sum()
        stop_pnl = results_df[results_df['exit_reason'] == 'stop']['pnl'].sum()
        close_pnl = results_df[results_df['exit_reason'] == 'close']['pnl'].sum()

    print(f"\n{'='*60}")
    print(f"汇总统计")
    print(f"{'='*60}")
    print(f"总交易次数: {total_trades}")
    print(f"盈利次数:   {wins}")
    print(f"亏损次数:   {losses}")
    print(f"胜率:       {wins/total_trades*100:.2f}%")
    print(f"总盈亏:     {total_pnl:.2f}")
    print(f"平均盈亏:   {avg_pnl:.2f}")
    print(f"平均盈亏%:  {avg_pnl_pct:.4f}%")
    print(f"最大盈利:   {max_win:.2f}")
    print(f"最大亏损:   {max_loss:.2f}")
    if 'exit_reason' in results_df.columns:
        print(f"\n出场方式:")
        print(f"  止损出场: {stop_trades}笔 ({stop_trades/total_trades*100:.1f}%)  "
              f"盈亏: {stop_pnl:.2f}")
        print(f"  收盘出场: {close_trades}笔 ({close_trades/total_trades*100:.1f}%)  "
              f"盈亏: {close_pnl:.2f}")

    # ---- 逐年表现 ----
    results_df['year'] = pd.to_datetime(results_df['date']).dt.year
    print(f"\n--- 逐年表现 ---")
    yearly = results_df.groupby('year').agg(
        交易次数=('pnl', 'count'),
        胜率=('pnl', lambda x: (x > 0).sum() / len(x) * 100),
        总盈亏=('pnl', 'sum'),
        平均盈亏=('pnl', 'mean'),
    )
    for year, row in yearly.iterrows():
        print(f"  {year}: {int(row['交易次数']):>4}笔  "
              f"胜率{row['胜率']:5.1f}%  "
              f"盈亏{row['总盈亏']:>10.2f}  均{row['平均盈亏']:>8.2f}")

    # ---- 品种表现 ----
    print(f"\n--- 品种表现 ---")
    sym_stats = results_df.groupby('symbol').agg(
        交易次数=('pnl', 'count'),
        胜率=('pnl', lambda x: (x > 0).sum() / len(x) * 100),
        总盈亏=('pnl', 'sum'),
    ).sort_values('总盈亏', ascending=False)
    for sym, row in sym_stats.iterrows():
        print(f"  {sym:<20} {int(row['交易次数']):>4}笔  "
              f"胜率{row['胜率']:5.1f}%  盈亏{row['总盈亏']:>10.2f}")

    # ---- 保存详细结果 ----
    results_df.to_csv(CONFIG['output_file'], index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存到 {CONFIG['output_file']}")


if __name__ == '__main__':
    main()
