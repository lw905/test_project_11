%% N型走势量化策略（日内回测）
% ============================
%
% 策略核心逻辑：
%   A(起涨低点) → B(第一波高点) → C(回调低点，不破A) → 突破B高点 → 开仓做多
%   止损设在A点低点，收盘强制平仓，每日独立，不持仓过夜。
%
% 数据格式：parquet 分钟线，字段包含 Datetime / Open / High / Low / Close / Volume
%
% 核心约束：不使用未来数据 —— 所有判断仅基于当前K线及之前的历史数据

clear; clc;

%% ============================================================
% 策略参数 — 所有可调参数集中在这里
% ============================================================
CONFIG = struct();

% 枢轴点检测：一根K线要成为"枢轴点"，需要比左右各 N 根K线的极值更高/更低
% 并且要等右侧 N 根K线走完才能确认，确保不用未来数据
CONFIG.pivot_lookback = 6;

% 第一段上涨(A→B)的最小幅度，小于此幅度的"上涨"视为噪音
CONFIG.min_leg_pct = 0.003;       % 0.3%

% 回调(B→C)幅度占第一段(A→B)的比例上限
% 例如 0.8 表示回调最多吃掉第一段涨幅的80%，必须保留至少20%
CONFIG.max_pullback_pct = 0.8;

% 回调(B→C)的最小幅度，必须实质性回调才构成N型
% 如果B之后几乎没有下跌就直接突破，不是N型，可能只是单边拉升
CONFIG.min_pullback_pct = 0.001;  % 0.1%

% 趋势过滤：日线收盘价 > N日均线时才允许做多，0=关闭过滤
CONFIG.trend_ma_period = 0;

% 回测起始日期
CONFIG.start_date = '2022-01-01';

% 数据目录与输出文件
CONFIG.data_dir = 'long';
CONFIG.target_symbol = 'IC9999.CCFX';
CONFIG.output_file = 'n_pattern_results.csv';

%% ============================================================
% 运行主程序
% ============================================================
main(CONFIG);

%% ============================================================
% 第1步：枢轴点检测
% ============================================================
function [pivot_highs, pivot_lows] = find_pivots(highs, lows, lookback)
    % 检测K线图中的局部高点和局部低点（枢轴点）。
    %
    % 原理：
    %   对于一根K线 i，如果它的最高价 > 左边 lookback 根K线的最高价
    %                         且 >= 右边 lookback 根K线的最高价
    %   则 i 被标记为「枢轴高点」。
    %
    %   同理，如果它的最低价 < 左边 lookback 根K线的最低价
    %                         且 <= 右边 lookback 根K线的最低价
    %   则 i 被标记为「枢轴低点」。
    %
    % 无未来数据保证：
    %   一根K线要等右侧 lookback 根K线走完才能确认，在此之前不标记。
    %
    % 输入：
    %   highs, lows: K线最高价和最低价向量
    %   lookback:    左右确认窗口大小
    %
    % 输出：
    %   pivot_highs, pivot_lows: logical 向量，true 表示该位置是枢轴点

    n = length(highs);
    pivot_highs = false(n, 1);
    pivot_lows  = false(n, 1);

    for i = (lookback + 1):(n - lookback)
        % --- 检测枢轴高点 ---
        left_max  = max(highs(i - lookback : i - 1));
        right_max = max(highs(i + 1 : i + lookback));
        if highs(i) > left_max && highs(i) >= right_max
            pivot_highs(i) = true;
        end

        % --- 检测枢轴低点 ---
        left_min  = min(lows(i - lookback : i - 1));
        right_min = min(lows(i + 1 : i + lookback));
        if lows(i) < left_min && lows(i) <= right_min
            pivot_lows(i) = true;
        end
    end
end

%% ============================================================
% 第2步：N型形态识别 + 交易信号生成
% ============================================================
function signals = find_n_pattern_signals(day_df, config)
    % 在单日分钟K线数据中扫描 N 型走势，生成做多信号。
    %
    % N型形态的时间顺序（从左到右）：
    %   A（起涨低点）→ B（第一波高点）→ C（回调低点，高于A）
    %   → 当前K线突破B ← 触发开仓信号
    %
    % 输入：
    %   day_df: 当天分钟K线 table（必须已按时间排序）
    %   config: 策略参数 struct
    %
    % 输出：
    %   signals: struct 数组，每个元素包含：
    %     {entry_time, entry_price, exit_time, exit_price, exit_reason, pnl, pnl_pct}

    lookback = config.pivot_lookback;
    min_leg = config.min_leg_pct;
    max_pullback = config.max_pullback_pct;

    n = height(day_df);
    highs  = day_df.High;
    lows   = day_df.Low;
    closes = day_df.Close;
    times  = day_df.Datetime;
    opens  = day_df.Open;

    % 先一次性对整个 day_df 做枢轴点检测
    [pivot_highs, pivot_lows] = find_pivots(highs, lows, lookback);

    % 已确认的枢轴点列表（按时间顺序追加）
    % 每行: [K线索引, 类型], 类型 'H'=高点, 'L'=低点
    confirmed_pivots = zeros(0, 2);  % [索引, 1=H/2=L]

    % 信号收集（用 cell 数组暂存，最后转 struct）
    signal_cells = {};  % 每行 {entry_time, entry_price, exit_time, exit_price, exit_reason, pnl, pnl_pct}

    % 持仓状态
    in_trade = false;
    entry_time = NaT;
    entry_price = NaN;
    stop_loss = NaN;

    % 最小起始索引：需要足够的历史K线才能找到枢轴点
    min_start_idx = lookback * 4 + 1;  % MATLAB 1-indexed

    for i = min_start_idx:n
        current_date = dateshift(times(i), 'start', 'day');

        % ========================================
        % 状态：已在持仓中
        % ========================================
        if in_trade
            exit_price = NaN;
            exit_reason = 'close';

            % --- 止损检查：当前K线最低价是否跌破A点 ---
            if lows(i) < stop_loss
                exit_price = stop_loss;
                exit_reason = 'stop';
            end

            % --- 收盘出场检查：是否是当天最后一根K线 ---
            is_last = false;
            if i == n
                is_last = true;
            else
                next_date = dateshift(times(i + 1), 'start', 'day');
                if next_date ~= current_date
                    is_last = true;
                end
            end

            % 如果先触发了止损就不再按收盘处理
            if ~strcmp(exit_reason, 'stop') && is_last
                exit_price = closes(i);
                exit_reason = 'close';
            end

            % 满足出场条件，记录这笔交易
            if ~isnan(exit_price)
                signal_cells{end + 1, 1} = entry_time;
                signal_cells{end, 2} = entry_price;
                signal_cells{end, 3} = times(i);
                signal_cells{end, 4} = exit_price;
                signal_cells{end, 5} = exit_reason;
                signal_cells{end, 6} = exit_price - entry_price;
                signal_cells{end, 7} = (exit_price - entry_price) / entry_price;
                in_trade = false;
            end
            continue;  % 持仓期间不做新的入场判断
        end

        % ========================================
        % 状态：空仓，寻找入场信号
        % ========================================

        % --- 确认刚刚成熟的枢轴点 ---
        confirm_idx = i - lookback;
        if confirm_idx > lookback && confirm_idx <= n - lookback
            if pivot_highs(confirm_idx)
                confirmed_pivots(end + 1, :) = [confirm_idx, 1];  % 1 = H
            end
            if pivot_lows(confirm_idx)
                confirmed_pivots(end + 1, :) = [confirm_idx, 2];  % 2 = L
            end
        end

        % 只保留最近的枢轴点
        if size(confirmed_pivots, 1) > 10
            confirmed_pivots = confirmed_pivots(end - 9:end, :);
        end

        % 至少需要3个枢轴点才能构成 L→H→L 序列
        if size(confirmed_pivots, 1) < 3
            continue;
        end

        % --- 从已确认枢轴点中倒序查找 N 型形态 ---
        % 按时间顺序：L(A) → H(B) → L(C)
        c_idx = 0;  b_idx = 0;  a_idx = 0;

        for j = size(confirmed_pivots, 1):-1:1
            idx = confirmed_pivots(j, 1);
            ptype = confirmed_pivots(j, 2);  % 1=H, 2=L
            if ptype == 2 && c_idx == 0          % 找到第一个L → C
                c_idx = idx;
            elseif ptype == 1 && c_idx > 0 && b_idx == 0 && idx < c_idx  % C之前H → B
                b_idx = idx;
            elseif ptype == 2 && b_idx > 0 && a_idx == 0 && idx < b_idx  % B之前L → A
                a_idx = idx;
                break;
            end
        end

        if a_idx == 0 || b_idx == 0 || c_idx == 0
            continue;
        end

        % ========================================
        % 验证 N 型的各个条件
        % ========================================

        a_low  = lows(a_idx);
        b_high = highs(b_idx);
        c_low  = lows(c_idx);

        % 条件1: B 必须高于 A（第一段确实是上涨）
        if b_high <= a_low, continue; end

        % 条件2: A→B 的涨幅必须达到最小阈值
        first_leg = (b_high - a_low) / a_low;
        if first_leg < min_leg, continue; end

        % 条件3: C 必须高于 A（回调不破起涨点）
        if c_low <= a_low, continue; end

        % 条件4: 回调幅度不能太大
        pullback_ratio = (b_high - c_low) / (b_high - a_low);
        if pullback_ratio > max_pullback, continue; end

        % 条件5: 回调幅度必须足够（实质回调）
        pullback_pct = (b_high - c_low) / b_high;
        if pullback_pct < config.min_pullback_pct, continue; end

        % 条件6: 当前K线突破 B 点高点（核心做多信号）
        if highs(i) <= b_high, continue; end

        % ========================================
        % 所有条件满足 → 触发做多信号！
        % ========================================
        entry_time = times(i);
        entry_price = max(b_high + 0.01, opens(i));
        stop_loss = a_low;
        in_trade = true;
    end

    % --- 将 cell 数组转为 struct 数组 ---
    if isempty(signal_cells)
        signals = [];
    else
        signals = cell2struct(signal_cells, ...
            {'entry_time', 'entry_price', 'exit_time', 'exit_price', ...
             'exit_reason', 'pnl', 'pnl_pct'}, 2);
    end
end

%% ============================================================
% 第3步：逐日回测
% ============================================================
function all_signals = backtest_symbol(filepath, config)
    % 对一个期货品种的完整分钟数据进行回测。
    %
    % 流程：
    %   1. 读取 parquet 文件，按时间排序
    %   2. 按交易日分组（每一天独立）
    %   3. 对每天的数据运行 N 型识别
    %   4. 汇总所有交易信号
    %
    % 趋势过滤（可选）：
    %   只有前一日收盘 > 均线的日子才允许做多。

    % 读取数据并排序
    df = parquetread(filepath);
    df = sortrows(df, 'Datetime');

    % 提取日期列（去掉时分秒）
    df.Date = dateshift(df.Datetime, 'start', 'day');
    all_dates = unique(df.Date);

    % ---- 构建日线并计算趋势均线 ----
    % 日线取每日最后一根K线的收盘价
    [G, date_groups] = findgroups(df.Date);
    daily_close = splitapply(@(x) x(end), df.Close, G);
    daily_date = date_groups;
    ma_period = config.trend_ma_period;

    % 如果启用了趋势过滤，预先计算哪些日期满足"前日收盘 > MA"
    allowed_dates = [];
    if ma_period > 0
        for idx = 1:length(daily_date)
            d = daily_date(idx);
            if d < datetime(config.start_date)
                continue;
            end
            if idx > ma_period
                ma = mean(daily_close(idx - ma_period : idx - 1));
                if daily_close(idx - 1) > ma
                    allowed_dates = [allowed_dates; d];
                end
            end
        end
    end

    % 收集所有信号
    all_signal_list = {};

    for d_idx = 1:length(all_dates)
        d = all_dates(d_idx);

        % 过滤起始日期
        if d < datetime(config.start_date)
            continue;
        end

        % 趋势过滤
        if ma_period > 0 && ~ismember(d, allowed_dates)
            continue;
        end

        % 提取当天数据
        day_rows = df.Date == d;
        day_df = df(day_rows, :);

        % 当天K线太少，跳过
        if height(day_df) < config.pivot_lookback * 4
            continue;
        end

        % 对该日运行 N 型识别
        signals = find_n_pattern_signals(day_df, config);

        if ~isempty(signals)
            % 添加品种和日期信息
            [~, sym_name, ~] = fileparts(filepath);
            for s = 1:length(signals)
                signals(s).symbol = sym_name;
                signals(s).date = d;
            end
            all_signal_list{end + 1} = signals;
        end
    end

    % ---- 合并所有信号 ----
    if isempty(all_signal_list)
        all_signals = [];
    else
        % 将所有 struct 数组合并
        all_signals = all_signal_list{1};
        for k = 2:length(all_signal_list)
            all_signals = [all_signals; all_signal_list{k}];
        end
    end
end

%% ============================================================
% 第4步：主程序
% ============================================================
function main(config)
    % 主流程：读取数据 → 回测 → 输出统计结果

    data_dir = config.data_dir;
    target = config.target_symbol;

    % 查找对应的 parquet 文件
    file_pattern = fullfile(data_dir, [target, '.parquet']);
    file_info = dir(file_pattern);

    if isempty(file_info)
        fprintf('未找到品种 %s\n', target);
        return;
    end

    filepath = fullfile(data_dir, file_info(1).name);

    % ---- 打印参数 ----
    if config.trend_ma_period > 0
        trend_str = sprintf('趋势均线=%d日MA', config.trend_ma_period);
    else
        trend_str = '趋势过滤=关闭';
    end
    fprintf('回测品种: %s\n', target);
    fprintf('参数: 枢轴窗口=%d, 最小涨幅=%.1f%%, 最大回调=%.0f%%, 最小回调=%.1f%%, %s\n\n', ...
        config.pivot_lookback, config.min_leg_pct * 100, config.max_pullback_pct * 100, ...
        config.min_pullback_pct * 100, trend_str);

    % ---- 运行回测 ----
    all_signals = backtest_symbol(filepath, config);

    if isempty(all_signals)
        fprintf('%s  无信号\n', target);
        fprintf('\n没有找到任何信号。尝试放宽参数…\n');
        return;
    end

    total_pnl = sum([all_signals.pnl]);
    wins = sum([all_signals.pnl] > 0);
    fprintf('%-20s 信号 %4d 个  胜率 %5.1f%%  总盈亏 %10.2f\n', ...
        target, length(all_signals), wins / length(all_signals) * 100, total_pnl);

    % ---- 汇总统计 ----
    total_trades = length(all_signals);
    wins_count = sum([all_signals.pnl] > 0);
    losses_count = sum([all_signals.pnl] <= 0);
    avg_pnl = mean([all_signals.pnl]);
    avg_pnl_pct = mean([all_signals.pnl_pct]) * 100;
    max_win = max([all_signals.pnl]);
    max_loss = min([all_signals.pnl]);

    % 出场方式统计
    exit_reasons = {all_signals.exit_reason};
    stop_trades = sum(strcmp(exit_reasons, 'stop'));
    close_trades = sum(strcmp(exit_reasons, 'close'));
    stop_pnl = sum([all_signals(strcmp(exit_reasons, 'stop')).pnl]);
    close_pnl = sum([all_signals(strcmp(exit_reasons, 'close')).pnl]);

    fprintf('\n%s\n', repmat('=', 1, 60));
    fprintf('汇总统计\n');
    fprintf('%s\n', repmat('=', 1, 60));
    fprintf('总交易次数: %d\n', total_trades);
    fprintf('盈利次数:   %d\n', wins_count);
    fprintf('亏损次数:   %d\n', losses_count);
    fprintf('胜率:       %.2f%%\n', wins_count / total_trades * 100);
    fprintf('总盈亏:     %.2f\n', total_pnl);
    fprintf('平均盈亏:   %.2f\n', avg_pnl);
    fprintf('平均盈亏%%:  %.4f%%\n', avg_pnl_pct);
    fprintf('最大盈利:   %.2f\n', max_win);
    fprintf('最大亏损:   %.2f\n', max_loss);

    fprintf('\n出场方式:\n');
    fprintf('  止损出场: %d笔 (%.1f%%)  盈亏: %.2f\n', ...
        stop_trades, stop_trades / total_trades * 100, stop_pnl);
    fprintf('  收盘出场: %d笔 (%.1f%%)  盈亏: %.2f\n', ...
        close_trades, close_trades / total_trades * 100, close_pnl);

    % ---- 逐年表现 ----
    dates_vec = datetime({all_signals.date}');
    years_vec = year(dates_vec);
    unique_years = unique(years_vec);

    fprintf('\n--- 逐年表现 ---\n');
    for y_idx = 1:length(unique_years)
        yr = unique_years(y_idx);
        yr_mask = years_vec == yr;
        yr_signals = all_signals(yr_mask);
        yr_count = length(yr_signals);
        yr_wins = sum([yr_signals.pnl] > 0);
        yr_total_pnl = sum([yr_signals.pnl]);
        yr_avg_pnl = mean([yr_signals.pnl]);
        fprintf('  %d: %4d笔  胜率%5.1f%%  盈亏%10.2f  均%8.2f\n', ...
            yr, yr_count, yr_wins / yr_count * 100, yr_total_pnl, yr_avg_pnl);
    end

    % ---- 保存详细结果到 CSV ----
    out_table = table();
    out_table.entry_time = datetime({all_signals.entry_time}');
    out_table.entry_price = [all_signals.entry_price]';
    out_table.exit_time = datetime({all_signals.exit_time}');
    out_table.exit_price = [all_signals.exit_price]';
    out_table.exit_reason = exit_reasons';
    out_table.pnl = [all_signals.pnl]';
    out_table.pnl_pct = [all_signals.pnl_pct]';
    out_table.symbol = {all_signals.symbol}';
    out_table.date = datetime({all_signals.date}');

    writetable(out_table, config.output_file);
    fprintf('\n详细结果已保存到 %s\n', config.output_file);
end
