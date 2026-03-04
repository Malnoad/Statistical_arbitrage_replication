# =============================================================================
# stat_arb_backtest_v5.py
# Statistical Arbitrage Backtest — 4 Strategies
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm

START_DATE = '1995-01-01'
END_DATE   = '2007-12-31'

print('Libraries loaded.')

# =============================================================================
# 1. LOAD DATA
# =============================================================================

# --- Stock prices ---
stock_df = pd.read_csv('sp500_adjclose.csv', index_col='Date', parse_dates=True)
print(f'Adj Close  shape: {stock_df.shape}')
print(f'Date range: {stock_df.index.min().date()} -> {stock_df.index.max().date()}')

# --- ETF prices ---
etf_df = pd.read_csv('sector_etfs_adjclose.csv', index_col='Date', parse_dates=True)
print(f'ETF shape: {etf_df.shape}')
print(f'Date range: {etf_df.index.min().date()} -> {etf_df.index.max().date()}')

# --- Market cap ---
mktcap_df = pd.read_csv('market_cap_daily.csv', index_col='Date', parse_dates=True)
print(f'Market cap shape: {mktcap_df.shape}')

# --- Sector mapping (from cache) ---
stock_map_etf = pd.read_csv("sector_mapping.csv")
print(stock_map_etf.head())

# --- SPY returns ---
spy_returns = pd.read_csv('spy_returns_1996_2007.csv')
spy_returns = spy_returns.set_index('Date')
spy_returns.index = pd.to_datetime(spy_returns.index)

# =============================================================================
# 2. CHECK DATA QUALITY
# =============================================================================

null_count = stock_df.isna().sum()
null_pct   = stock_df.isna().mean() * 100
zero_count = (stock_df == 0).sum()
neg_count  = (stock_df < 0).sum()

print(f'\nTotal trading days: {len(stock_df)}')
print(f'Stocks with >50% missing: {(null_pct > 50).sum()}')
print(f'Stocks with >20% missing: {(null_pct > 20).sum()}')
print(f'Stocks with >5% missing: {(null_pct > 5).sum()}')
print(f'Stocks fully empty (100% null): {(null_pct == 100).sum()}')
print(f'Any negative prices: {(neg_count > 0).sum()} tickers')
print(f'Any zero prices: {(zero_count > 0).sum()} tickers')

stock_qc = pd.DataFrame({
    'null_days':  null_count,
    'null_pct':   null_pct.round(1),
    'zero_price': zero_count,
    'neg_price':  neg_count,
}).sort_values('null_pct', ascending=False)
print(stock_qc.head(10))

# --- Missing per day plot ---
missing_per_day = stock_df.isna().sum(axis=1)
plt.plot(missing_per_day.index, missing_per_day.values, linewidth=0.7, color='tomato')
plt.title('# Tickers with Missing Price per Day')
plt.xlabel('Date')
plt.ylabel('# Tickers')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout()
plt.show()

# --- ETF quality ---
etf_summary = []
for etf in etf_df.columns:
    etf_summary.append({
        'ticker':      etf,
        'rows':        len(etf_df[etf].dropna()),
        'start':       etf_df[etf].first_valid_index(),
        'end':         etf_df[etf].last_valid_index(),
        'missing':     etf_df[etf].isna().sum(),
        'pct_missing': etf_df[etf].isna().mean() * 100,
        'negative':    (etf_df[etf] < 0).sum(),
        'duplicates':  etf_df[etf].index.duplicated().sum()
    })
etf_summary_df = pd.DataFrame(etf_summary)
print(etf_summary_df)

# --- Market cap quality ---
null_pct_mc = mktcap_df.isna().mean() * 100
print(f'Shape: {mktcap_df.shape}')
print(f'Date range: {mktcap_df.index.min().date()} -> {mktcap_df.index.max().date()}')
print(f'\nTickers with full market cap data (0% null): {(null_pct_mc == 0).sum()}')
print(f'Tickers completely missing: {(null_pct_mc == 100).sum()}')
print(f'Tickers with >50% missing: {(null_pct_mc > 50).sum()}')
print(f'Tickers with >20% missing: {(null_pct_mc > 20).sum()}')
print(f'Tickers with >5% missing: {(null_pct_mc > 5).sum()}')

# =============================================================================
# 3. CLEAN STOCK DATA
# =============================================================================

# Drop columns with > 5% missing
stock_cleaned = stock_df.loc[:, null_pct <= 5]
print("Original columns:", stock_df.shape[1])
print("Remaining columns:", stock_cleaned.shape[1])
print("Dropped columns:", stock_df.shape[1] - stock_cleaned.shape[1])

print("Total missing before:", stock_df.isna().sum().sum())
print("Total missing after :", stock_cleaned.isna().sum().sum())

# Remaining tickers with missing values
missing_per_col = stock_cleaned.isna().sum()
print(missing_per_col[missing_per_col > 0].sort_values(ascending=False))

# Drop remaining problem tickers
cols_to_drop = ["FCX", "WAB", "RMD", "DRI", "COR", "DLTR", "EME"]
stock_cleaned = stock_cleaned.drop(columns=cols_to_drop, errors='ignore')
print("Total missing before:", stock_df.isna().sum().sum())
print("Total missing after :", stock_cleaned.isna().sum().sum())

# =============================================================================
# 4. COMPUTE RETURNS
# =============================================================================

returns_all = stock_cleaned.pct_change().dropna()
etf_returns = etf_df.pct_change()

# =============================================================================
# 5. FILTER STOCKS BY MARKET CAP > $1B
# =============================================================================

cap_threshold = 1e9

common_cols        = stock_cleaned.columns.intersection(mktcap_df.columns)
returns_aligned    = returns_all[common_cols]
market_cap_aligned = mktcap_df[common_cols]
trade_mask         = market_cap_aligned.shift(1) > cap_threshold
returns_filtered   = returns_aligned.where(trade_mask)

print("Total stocks in model universe:", returns_filtered.shape[1])

# =============================================================================
# 6. SYNTHETIC ETFs
# =============================================================================

sector_map = {
    "Internet":               "HHH",
    "Real Estate":            "IYR",
    "Transportation":         "IYT",
    "Oil Exploration":        "OIH",
    "Regional Banks":         "RKH",
    "Retail":                 "RTH",
    "Semiconductors":         "SMH",
    "Utility":                "UTH",
    "Energy":                 "XLE",
    "Financial":              "XLF",
    "Industrial":             "XLI",
    "Technology":             "XLK",
    "Consumer Staples":       "XLP",
    "Healthcare":             "XLV",
    "Consumer discretionary": "XLY"
}

def build_synthetic_etfs(stock_map_etf, returns, market_cap_aligned):
    sector_returns = {}
    for sector in stock_map_etf["Sector_Classification"].unique():
        tickers = stock_map_etf.loc[
            stock_map_etf["Sector_Classification"] == sector, "Ticker"
        ]
        tickers = [t for t in tickers if t in returns.columns]
        if not tickers:
            continue
        mc      = market_cap_aligned[tickers].shift(1)
        mc      = mc.where(mc > 1e9)
        weights = mc.div(mc.sum(axis=1), axis=0)
        sec_ret = (weights * returns[tickers]).sum(axis=1)
        sector_returns[sector] = sec_ret
    return pd.DataFrame(sector_returns)


sector_returns = build_synthetic_etfs(
    stock_map_etf,
    returns=returns_filtered,
    market_cap_aligned=market_cap_aligned
)
sector_returns = sector_returns.rename(columns=sector_map)

print("Synthetic ETFs shape:", sector_returns.shape)
print("Sectors:", list(sector_returns.columns))
print(sector_returns.head())

# --- Compare synthetic vs actual ETFs ---
common_etfs = sector_returns.columns.intersection(etf_df.columns)
print("Common ETFs:", list(common_etfs))
for etf in common_etfs:
    plt.figure(figsize=(10, 5))
    plt.plot(sector_returns.index, sector_returns[etf], label=f"Synthetic {etf}", linewidth=2)
    plt.plot(etf_returns.index, etf_returns[etf], label=f"Actual {etf}", linewidth=2, alpha=0.7)
    plt.title(f"Synthetic vs Actual ETF: {etf}")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# --- Ticker → ETF map ---
ticker_to_etf = dict(zip(stock_map_etf["Ticker"], stock_map_etf["Sector_ETF"]))

# =============================================================================
# 7. HELPER FUNCTIONS
# =============================================================================

# --- Rolling residuals ---
def _rolling_residuals(stock_returns, factor_returns, stock_to_factor, window=60):
    idx            = stock_returns.index.intersection(factor_returns.index)
    stock_returns  = stock_returns.loc[idx]
    factor_returns = factor_returns.loc[idx]
    n              = len(idx)
    residuals      = pd.DataFrame(np.nan, index=idx, columns=stock_returns.columns)
    for ticker in stock_returns.columns:
        factor_col = stock_to_factor.get(ticker)
        if factor_col not in factor_returns.columns:
            continue
        r   = stock_returns[ticker].values
        f   = factor_returns[factor_col].values
        res = np.full(n, np.nan)
        for t in range(window, n):
            r_win = r[t - window: t]
            f_win = f[t - window: t]
            mask  = ~(np.isnan(r_win) | np.isnan(f_win))
            if mask.sum() < 2:
                continue
            X          = np.column_stack([np.ones(mask.sum()), f_win[mask]])
            coeffs, *_ = np.linalg.lstsq(X, r_win[mask], rcond=None)
            beta0, beta = coeffs
            if not (np.isnan(r[t]) or np.isnan(f[t])):
                res[t] = r[t] - beta0 - beta * f[t]
        residuals[ticker] = res
    return residuals

def compute_residuals_synthetic(stock_returns, sector_returns, stock_to_sector, window=60):
    return _rolling_residuals(stock_returns, sector_returns, stock_to_sector, window)

def compute_residuals_actual(stock_returns, etf_returns, stock_to_etf, window=60):
    return _rolling_residuals(stock_returns, etf_returns, stock_to_etf, window)


# --- OU fitting ---
def fit_ou(residuals, window=60, kappa_min=8.4):
    kappa    = pd.DataFrame(index=residuals.index, columns=residuals.columns, dtype=float)
    mu       = kappa.copy()
    sigma_eq = kappa.copy()
    for t in range(window, len(residuals)):
        date       = residuals.index[t]
        eps_window = residuals.iloc[t - window + 1:t + 1]
        for ticker in residuals.columns:
            eps = eps_window[ticker].dropna().values
            if len(eps) < window:
                continue
            x     = np.cumsum(eps)
            x_lag = x[:-1]
            x_now = x[1:]
            if len(x_lag) < 2:
                continue
            x_lag_mean = x_lag.mean()
            x_now_mean = x_now.mean()
            num   = np.sum((x_lag - x_lag_mean) * (x_now - x_now_mean))
            denom = np.sum((x_lag - x_lag_mean) ** 2)
            if denom <= 1e-12:
                continue
            b = num / denom
            if b <= 0 or b >= 1:
                continue
            k_annual = -np.log(b) * 252
            if k_annual < kappa_min:
                continue
            a        = x_now_mean - b * x_lag_mean
            m        = a / (1 - b)
            zeta     = x_now - (a + b * x_lag)
            var_zeta = np.var(zeta, ddof=1)
            if var_zeta <= 1e-12:
                continue
            kappa.loc[date, ticker]    = k_annual
            mu.loc[date, ticker]       = m
            sigma_eq.loc[date, ticker] = np.sqrt(var_zeta / (1 - b ** 2))
        valid = mu.loc[date].notna()
        if valid.any():
            mu.loc[date, valid] -= mu.loc[date, valid].mean()
    return kappa, mu, sigma_eq


# --- S-score ---
def compute_s_score(residuals, mu, sigma_eq, window=60):
    s = pd.DataFrame(index=residuals.index, columns=residuals.columns, dtype=float)
    for t in range(window, len(residuals)):
        date       = residuals.index[t]
        eps_window = residuals.iloc[t - window + 1:t + 1]
        X_t        = eps_window.cumsum().iloc[-1]
        m          = mu.loc[date]
        sig        = sigma_eq.loc[date]
        valid      = m.notna() & sig.notna() & X_t.notna() & (sig > 1e-12)
        s.loc[date, valid] = (X_t[valid] - m[valid]) / sig[valid]
    return s


# --- Signal generation ---
def generate_signals(s, s_open=1.25, s_close_long=0.50, s_close_short=0.75):
    signals = pd.DataFrame(0, index=s.index, columns=s.columns)
    current = pd.Series(0, index=s.columns)
    for t in range(1, len(s)):
        s_prev  = s.iloc[t - 1]
        new_pos = current.copy()
        for stock in s.columns:
            val = s_prev[stock]
            if pd.isna(val):
                continue
            if current[stock] == 0:
                if val < -s_open:
                    new_pos[stock] = 1
                elif val > s_open:
                    new_pos[stock] = -1
            elif current[stock] == 1:
                if val > -s_close_long:
                    new_pos[stock] = 0
            elif current[stock] == -1:
                if val < s_close_short:
                    new_pos[stock] = 0
        signals.iloc[t] = new_pos
        current         = new_pos
    return signals


# --- Portfolio construction ---
def construct_portfolio(signals, leverage=2.0, max_positions=100):
    weights         = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
    current_weights = pd.Series(0.0, index=signals.columns)
    t_size          = leverage / max_positions
    for date in signals.index:
        signal_today = signals.loc[date]
        new_weights  = current_weights.copy()
        for stock in signals.columns:
            sig    = signal_today[stock]
            prev_w = current_weights[stock]
            if   sig ==  1 and prev_w == 0: new_weights[stock] =  t_size
            elif sig == -1 and prev_w == 0: new_weights[stock] = -t_size
            elif sig ==  0 and prev_w != 0: new_weights[stock] =  0.0
        weights.loc[date] = new_weights
        current_weights   = new_weights
    return weights


# --- Backtest core ---
def backtest_core(weights_stock, weights_hedge, stock_returns, hedge_returns,
                  initial_equity=100, slippage=0.0005):
    common = (weights_stock.index
              .intersection(stock_returns.index)
              .intersection(hedge_returns.index))
    ws = weights_stock.loc[common].fillna(0)
    wh = weights_hedge.loc[common].fillna(0)
    rs = stock_returns.loc[common].fillna(0)
    rh = hedge_returns.loc[common].fillna(0)
    equity      = pd.Series(index=common, dtype=float)
    equity.iloc[0] = initial_equity
    for t in range(1, len(common)):
        E_prev    = equity.iloc[t - 1]
        Q_stock   = E_prev * ws.iloc[t]
        Q_hedge   = E_prev * wh.iloc[t]
        pnl_stock = (Q_stock * rs.iloc[t]).sum()
        pnl_hedge = (Q_hedge * rh.iloc[t]).sum()
        if t > 1:
            E_prev2      = equity.iloc[t - 2]
            Q_prev_stock = E_prev2 * ws.iloc[t - 1]
            Q_prev_hedge = E_prev2 * wh.iloc[t - 1]
        else:
            Q_prev_stock = pd.Series(0.0, index=ws.columns)
            Q_prev_hedge = pd.Series(0.0, index=wh.columns)
        cost = ((Q_stock - Q_prev_stock).abs().sum() +
                (Q_hedge - Q_prev_hedge).abs().sum()) * slippage
        equity.iloc[t] = E_prev + pnl_stock + pnl_hedge - cost
    return equity


def backtest_synthetic(weights, stock_returns, spy_returns,
                       window=60, initial_equity=100, slippage=0.0005):
    if isinstance(spy_returns, pd.DataFrame):
        spy_returns = spy_returns.iloc[:, 0]
    common        = (weights.index
                     .intersection(stock_returns.index)
                     .intersection(spy_returns.index))
    weights       = weights.loc[common].fillna(0)
    stock_returns = stock_returns.loc[common].fillna(0)
    spy_returns   = spy_returns.loc[common].fillna(0)
    port_ret      = (weights * stock_returns).sum(axis=1)
    cov           = port_ret.rolling(window).cov(spy_returns)
    var           = spy_returns.rolling(window).var().replace(0, np.nan)
    beta          = (cov / var).fillna(0)
    hedge_weights = pd.DataFrame(-beta.values, index=common, columns=["SPY"])
    hedge_returns = pd.DataFrame(spy_returns.values, index=common, columns=["SPY"])
    return backtest_core(weights, hedge_weights, stock_returns, hedge_returns,
                         initial_equity, slippage)


def compute_industry_hedge(weights, stock_to_etf):
    etf_map = pd.Series(stock_to_etf).reindex(weights.columns)
    return weights.T.groupby(etf_map).sum().T * -1


def backtest_actual(weights, stock_returns, etf_returns, stock_to_etf,
                    initial_equity=100, slippage=0.0005):
    common        = (weights.index
                     .intersection(stock_returns.index)
                     .intersection(etf_returns.index))
    weights       = weights.loc[common].fillna(0)
    stock_returns = stock_returns.loc[common].fillna(0)
    etf_returns   = etf_returns.loc[common].fillna(0)
    hedge_weights = compute_industry_hedge(weights, stock_to_etf)
    hedge_weights = hedge_weights.reindex(columns=etf_returns.columns).fillna(0)
    return backtest_core(weights, hedge_weights, stock_returns, etf_returns,
                         initial_equity, slippage)


# --- PCA helpers ---
def standardize_window(window_data):
    mean_i = window_data.mean(axis=0)
    std_i  = window_data.std(axis=0, ddof=1).replace(0, np.nan)
    Y      = (window_data - mean_i) / std_i
    return Y.dropna(axis=1)

def compute_empirical_correlation(Y):
    M = Y.shape[0]
    return (Y.values.T @ Y.values) / (M - 1)

def eigen_decomposition_sorted(corr):
    eigvals, eigvecs = np.linalg.eigh(corr)
    idx = np.argsort(eigvals)[::-1]
    return eigvals[idx], eigvecs[:, idx]

def rolling_pca_engine(returns, window=252, mode="fixed", n_factors=15, variance_cutoff=0.55):
    dates      = returns.index
    eigen_dict = {}
    k_series   = {}
    for t in range(window, len(dates)):
        date_t      = dates[t]
        window_data = returns.iloc[t - window + 1:t + 1].dropna(axis=1)
        if window_data.shape[1] < 2:
            continue
        Y = standardize_window(window_data)
        if Y.shape[1] < 2:
            continue
        stocks             = Y.columns
        corr               = compute_empirical_correlation(Y)
        eigvals, eigvecs   = eigen_decomposition_sorted(corr)
        if mode == "fixed":
            k = min(n_factors, len(stocks) - 1)
        elif mode == "variable":
            cumvar = np.cumsum(eigvals) / eigvals.sum()
            k      = int(np.searchsorted(cumvar, variance_cutoff)) + 1
            k      = min(k, len(stocks) - 1)
            k_series[date_t] = k
        else:
            raise ValueError("mode must be 'fixed' or 'variable'")
        V     = eigvecs[:, :k]
        std_i = window_data[stocks].std(ddof=1).values
        std_i[std_i == 0] = np.nan
        Q = V / std_i.reshape(-1, 1)
        eigen_dict[date_t] = {"Q": Q, "V": V, "k": k, "stocks": stocks, "std_i": std_i}
    if mode == "fixed":
        return eigen_dict
    else:
        return eigen_dict, pd.Series(k_series)

def compute_pca_residuals(returns, eigen_dict, reg_window=60):
    dates     = returns.index
    residuals = pd.DataFrame(index=dates, columns=returns.columns, dtype=float)
    for date_t in sorted(eigen_dict.keys()):
        if date_t not in dates:
            continue
        t = dates.get_loc(date_t)
        if t < reg_window:
            continue
        info      = eigen_dict[date_t]
        Q         = info["Q"]
        stocks    = info["stocks"]
        reg_dates = dates[t - reg_window + 1:t + 1]
        R_arr     = returns.loc[reg_dates, stocks].fillna(0).values
        F_win     = R_arr @ Q
        r_today   = returns.loc[date_t, stocks].fillna(0).values
        f_today   = r_today @ Q
        if np.isnan(f_today).any():
            continue
        X       = np.column_stack([np.ones(reg_window), F_win])
        x_today = np.concatenate([[1.0], f_today])
        XtX_inv = np.linalg.pinv(X.T @ X)
        for i, stock in enumerate(stocks):
            y                             = R_arr[:, i]
            beta_full                     = XtX_inv @ (X.T @ y)
            fitted                        = x_today @ beta_full
            residuals.loc[date_t, stock]  = returns.loc[date_t, stock] - fitted
    return residuals

def backtest_pca_strategy(weights, returns, spy_returns,
                           initial_equity=100, slippage=0.0005, beta_window=60):
    spy_returns = spy_returns.squeeze()
    common      = (weights.index
                   .intersection(returns.index)
                   .intersection(spy_returns.index))
    weights     = weights.loc[common].fillna(0)
    returns     = returns.loc[common].fillna(0)
    spy_returns = spy_returns.loc[common].fillna(0)
    stock_ret   = (weights * returns).sum(axis=1)
    rolling_cov = stock_ret.rolling(beta_window).cov(spy_returns)
    rolling_var = spy_returns.rolling(beta_window).var().replace(0, np.nan)
    beta_series = (rolling_cov / rolling_var).fillna(0).replace([np.inf, -np.inf], 0)
    w           = weights.fillna(0)
    bh          = beta_series.fillna(0)
    equity      = pd.Series(index=common, dtype=float)
    equity.iloc[0] = initial_equity
    for t in range(1, len(common)):
        E         = equity.iloc[t - 1]
        Q_stock   = E * w.iloc[t]
        Q_spy     = -E * bh.iloc[t]
        pnl_stock = (Q_stock * returns.iloc[t]).sum()
        pnl_spy   = Q_spy * spy_returns.iloc[t]
        if t > 1:
            E_prev2      = equity.iloc[t - 2]
            Q_prev_stock = E_prev2 * w.iloc[t - 1]
            Q_prev_spy   = -E_prev2 * bh.iloc[t - 1]
        else:
            Q_prev_stock = pd.Series(0.0, index=weights.columns)
            Q_prev_spy   = 0.0
        cost = ((Q_stock - Q_prev_stock).abs().sum() +
                abs(Q_spy - Q_prev_spy)) * slippage
        equity.iloc[t] = E + pnl_stock + pnl_spy - cost
    return equity


# --- Performance metrics ---
def total_return(equity):
    equity = equity.dropna()
    return equity.iloc[-1] / equity.iloc[0] - 1

def annual_sharpe_table(equity, trading_days=252, rf_rate=0.0):
    equity  = equity.dropna()
    returns = equity.pct_change().dropna()
    sharpe_list = []
    for year, ret_year in returns.groupby(returns.index.year):
        if len(ret_year) < 30:
            continue
        annual_return = (1 + ret_year).prod() - 1
        annual_vol    = ret_year.std() * np.sqrt(trading_days)
        sharpe        = (annual_return - rf_rate) / annual_vol if annual_vol > 0 else np.nan
        sharpe_list.append({"Year": year, "Annual Sharpe Ratio": sharpe})
    return pd.DataFrame(sharpe_list).set_index("Year")

def compute_annual_sharpe(daily_returns, trading_days=252):
    daily_returns       = daily_returns.copy()
    daily_returns.index = pd.to_datetime(daily_returns.index)
    sharpe_dict         = {}
    for year in sorted(daily_returns.index.year.unique()):
        data = daily_returns[daily_returns.index.year == year].dropna()
        if len(data) < 30 or data.std() == 0:
            sharpe_dict[year] = np.nan
            continue
        sharpe_dict[year] = np.sqrt(trading_days) * data.mean() / data.std()
    return pd.Series(sharpe_dict)

def compute_total_sharpe(daily_returns, trading_days=252):
    daily_returns = daily_returns.dropna()
    if len(daily_returns) < 30 or daily_returns.std() == 0:
        return np.nan
    return np.sqrt(trading_days) * daily_returns.mean() / daily_returns.std()

def performance_report(equity_series):
    daily_ret     = equity_series.pct_change().fillna(0)
    annual_sharpe = compute_annual_sharpe(daily_ret)
    total_sharpe  = compute_total_sharpe(daily_ret)
    tot_return    = equity_series.iloc[-1] / equity_series.iloc[0] - 1
    print("\n========== PERFORMANCE ==========")
    print(f"Total Return: {tot_return*100:.2f}%")
    print(f"Total Sharpe: {total_sharpe:.2f}")
    print("\nSharpe by Year:")
    print(annual_sharpe)
    return annual_sharpe, total_sharpe


# =============================================================================
# 8. STRATEGY: SYNTHETIC ETFs
# =============================================================================

residuals_syn = compute_residuals_synthetic(
    returns_filtered, sector_returns, ticker_to_etf, window=60)

kappa_syn, mu_syn, sigma_eq_syn = fit_ou(residuals_syn, window=60)

s_syn       = compute_s_score(residuals_syn, mu_syn, sigma_eq_syn, window=60)
signals_syn = generate_signals(s_syn)
weights_syn = construct_portfolio(signals_syn, leverage=2.0, max_positions=100)
equity_syn  = backtest_synthetic(weights_syn, returns_filtered, spy_returns, window=60)

print("=== Synthetic ETFs ===")
print("Total Return:", total_return(equity_syn))
print("\nAnnual Sharpe:")
print(annual_sharpe_table(equity_syn))

plt.figure()
plt.plot(equity_syn)
plt.title("Historical PnL of Strategy (Synthetic ETFs)")
plt.xlabel("Date")
plt.ylabel("Cumulative PnL (Base = 100)")
plt.show()

# =============================================================================
# 9. STRATEGY: ACTUAL ETFs
# =============================================================================

residuals_act = compute_residuals_actual(
    returns_filtered, etf_returns, ticker_to_etf, window=60)

kappa_act, mu_act, sigma_eq_act = fit_ou(residuals_act, window=60)

s_act       = compute_s_score(residuals_act, mu_act, sigma_eq_act, window=60)
signals_act = generate_signals(s_act)
weights_act = construct_portfolio(signals_act, leverage=2.0, max_positions=100)
equity_act  = backtest_actual(weights_act, returns_filtered, etf_returns, ticker_to_etf)

print("=== Actual ETFs ===")
print("Total Return:", total_return(equity_act))
print("\nAnnual Sharpe:")
print(annual_sharpe_table(equity_act))

plt.figure()
plt.plot(equity_act)
plt.title("Historical PnL of Strategy (Actual ETFs)")
plt.xlabel("Date")
plt.ylabel("Cumulative PnL (Base = 100)")
plt.show()

# --- Compare Synthetic vs Actual ---
common_index = equity_syn.index.intersection(equity_act.index)
plt.figure(figsize=(12, 6))
plt.plot(equity_syn.loc[common_index], label="Synthetic ETFs")
plt.plot(equity_act.loc[common_index], label="Actual ETFs")
plt.title("Equity Curve Comparison")
plt.xlabel("Date")
plt.ylabel("Equity Value")
plt.legend()
plt.grid(True)
plt.show()

# =============================================================================
# 10. STRATEGY: PCA FIXED (15 factors)
# =============================================================================

eigen_fixed = rolling_pca_engine(returns_filtered, window=252, mode="fixed", n_factors=15)
resid_fixed = compute_pca_residuals(returns_filtered, eigen_fixed, reg_window=60)

kappa_fixed, mu_fixed, sigma_eq_fixed = fit_ou(resid_fixed, window=60)

s_fixed       = compute_s_score(resid_fixed, mu_fixed, sigma_eq_fixed, window=60)
signals_fixed = generate_signals(s_fixed)
weights_fixed = construct_portfolio(signals_fixed, leverage=2.0, max_positions=100)
equity_fixed  = backtest_pca_strategy(weights_fixed, returns_filtered, spy_returns)

print("=== PCA Fixed (15 factors) ===")
print("Total Return  :", total_return(equity_fixed))
print("\nAnnual Sharpe :")
print(annual_sharpe_table(equity_fixed))

performance_report(equity_fixed)

plt.figure(figsize=(10, 5))
plt.plot(equity_fixed)
plt.title("PCA Fixed Variable Strategy Equity Curve")
plt.show()

# =============================================================================
# 11. STRATEGY: PCA VARIABLE (55% variance)
# =============================================================================

eigen_var, k_series = rolling_pca_engine(
    returns_filtered, window=252, mode="variable", variance_cutoff=0.55)
resid_var = compute_pca_residuals(returns_filtered, eigen_var, reg_window=60)

kappa_var, mu_var, sigma_eq_var = fit_ou(resid_var, window=60)

s_var       = compute_s_score(resid_var, mu_var, sigma_eq_var, window=60)
signals_var = generate_signals(s_var)
weights_var = construct_portfolio(signals_var, leverage=2.0, max_positions=100)
equity_var  = backtest_pca_strategy(weights_var, returns_filtered, spy_returns)

print("=== PCA Variable (55% variance) ===")
print("Total Return  :", total_return(equity_var))
print("\nAnnual Sharpe :")
print(annual_sharpe_table(equity_var))

performance_report(equity_var)

plt.figure(figsize=(10, 5))
plt.plot(equity_var)
plt.title("PCA - 55% explained variance Strategy Equity Curve")
plt.show()

# =============================================================================
# 12. COMBINED CHARTS
# =============================================================================

# Fixed vs Variable PnL
common_index = equity_fixed.index.intersection(equity_var.index)
eq_fixed     = equity_fixed.loc[common_index]
eq_variable  = equity_var.loc[common_index]
pnl_fixed    = eq_fixed - eq_fixed.iloc[0]
pnl_variable = eq_variable - eq_variable.iloc[0]

plt.figure(figsize=(12, 6))
plt.plot(pnl_variable, linestyle='--', label="55% Explained Variance PCA")
plt.plot(pnl_fixed,    linestyle='-',  label="PCA using 15 Eigenportfolios")
plt.title("Comparison of the PNLs for the Fixed (55%) PCA and 15 PCA Strategy")
plt.xlabel("Date")
plt.ylabel("Cumulative PnL")
plt.legend()
plt.grid(True)
plt.show()

# All 4 strategies — Equity curves
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(equity_fixed.index, equity_fixed.values, label='Fixed (15 Factors)',     linewidth=2, color='blue')
ax.plot(equity_var.index,   equity_var.values,   label='Variable (55% Variance)', linewidth=2, color='red')
ax.plot(equity_syn.index,   equity_syn.values,   label='Synthetic ETFs',          linewidth=2, color='green')
ax.plot(equity_act.index,   equity_act.values,   label='Actual ETFs',             linewidth=2, color='orange')
ax.set_title('Equity Curves Comparison - 4 Strategies', fontsize=14, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Equity Value', fontsize=12)
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Summary
print("=" * 70)
print("EQUITY CURVES SUMMARY")
print("=" * 70)
for name, eq in [("Fixed (15 Factors)", equity_fixed),
                  ("Variable (55% Variance)", equity_var),
                  ("Synthetic ETFs", equity_syn),
                  ("Actual ETFs", equity_act)]:
    print(f"\n{name}:")
    print(f"  Initial: {eq.iloc[0]:>10.2f}")
    print(f"  Final:   {eq.iloc[-1]:>10.2f}")
    print(f"  Return:  {(eq.iloc[-1]/eq.iloc[0]-1)*100:>9.2f}%")
print("=" * 70)

# All 4 strategies — PnL curves
initial_capital = 100
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot((equity_fixed - initial_capital).index, (equity_fixed - initial_capital).values,
        label='Fixed (15 Factors)',     linewidth=2, color='blue')
ax.plot((equity_var - initial_capital).index,   (equity_var - initial_capital).values,
        label='Variable (55% Variance)', linewidth=2, color='red')
ax.plot((equity_syn - initial_capital).index,   (equity_syn - initial_capital).values,
        label='Synthetic ETFs',          linewidth=2, color='green')
ax.plot((equity_act - initial_capital).index,   (equity_act - initial_capital).values,
        label='Actual ETFs',             linewidth=2, color='orange')
ax.set_title('PnL Comparison - 4 Strategies', fontsize=14, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Cumulative PnL', fontsize=12)
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# =============================================================================
# 13. SHARPE RATIO TABLE BY YEAR — ALL 4 STRATEGIES
# =============================================================================

def compute_sharpe_table(equity_curve, weights, sector_returns_data, trading_days=252):
    portfolio_ret = equity_curve.pct_change().dropna()
    common_idx    = portfolio_ret.index.intersection(sector_returns_data.index)
    portfolio_ret = portfolio_ret.loc[common_idx]
    sector_ret    = sector_returns_data.loc[common_idx]
    years         = sorted(portfolio_ret.index.year.unique())
    sharpe_dict   = {}
    for year in years:
        sharpe_dict[year] = {}
        year_mask         = portfolio_ret.index.year == year
        port_ret_year     = portfolio_ret.loc[year_mask]
        sector_ret_year   = sector_ret.loc[year_mask]
        for etf_col in sector_ret_year.columns:
            ret = sector_ret_year[etf_col].dropna()
            if len(ret) < 20:
                sharpe_dict[year][etf_col] = np.nan
                continue
            std = ret.std()
            sharpe_dict[year][etf_col] = (ret.mean() / std * np.sqrt(trading_days)
                                          if std > 1e-10 else np.nan)
        ret = port_ret_year.dropna()
        if len(ret) >= 20:
            std = ret.std()
            sharpe_dict[year]['Portfolio'] = (ret.mean() / std * np.sqrt(trading_days)
                                              if std > 1e-10 else np.nan)
        else:
            sharpe_dict[year]['Portfolio'] = np.nan
    sharpe_df           = pd.DataFrame(sharpe_dict).T
    sharpe_df.index.name = 'Year'
    return sharpe_df.round(3)


print("\n" + "=" * 100)
print("SHARPE RATIO BY YEAR - 4 STRATEGIES")
print("=" * 100)

print("\n1. FIXED (15 Factors):")
print("-" * 100)
sharpe_fixed = compute_sharpe_table(equity_fixed, weights_fixed, sector_returns)
print(sharpe_fixed)

print("\n\n2. VARIABLE (55% Variance):")
print("-" * 100)
sharpe_var = compute_sharpe_table(equity_var, weights_var, sector_returns)
print(sharpe_var)

print("\n\n3. SYNTHETIC ETFs:")
print("-" * 100)
sharpe_syn = compute_sharpe_table(equity_syn, weights_syn, sector_returns)
print(sharpe_syn)

print("\n\n4. ACTUAL ETFs:")
print("-" * 100)
sharpe_act = compute_sharpe_table(equity_act, weights_act, etf_returns)
print(sharpe_act)

print("\n" + "=" * 100)
