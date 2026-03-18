# =============================================================================
# stat_arb_backtest_v5.py
# Statistical Arbitrage Backtest — 4 Strategies
# =============================================================================

from curses import window

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

START_DATE = '2018-01-02'
END_DATE   = '2025-12-31'

print('Libraries loaded.')

# =============================================================================
# 1. LOAD DATA
# =============================================================================

# --- Stock prices ---
prices_df = pd.read_csv('vietnam_stocks_price.csv', index_col= 0, parse_dates=True)
print(f'Adj Close  shape: {prices_df.shape}')
print(f'Date range: {prices_df.index.min().date()} -> {prices_df.index.max().date()}')

# --- Stock volumes ---
volume_df = pd.read_csv('vietnam_stocks_volume.csv', index_col= 0, parse_dates=True)
print(f'Volume shape: {volume_df.shape}')

# --- FU VN30 ---
fu_df = pd.read_csv('fu_vn30_price.csv', index_col=0, parse_dates=True)
print(f'FU VN30 shape: {fu_df.shape}')


# =============================================================================
# FUNCTIONS
# =============================================================================


# 1. Data preparation and universe selection
# --- Function to select stocks with highest liquidity ---
def top_liquidity_stocks(price_df, volume_df, top_n=300, average_window=60):
    # Calculate liquidity metrics (average volume* price over lookback window)
    price_win = price_df.tail(average_window)
    volume_win = volume_df.tail(average_window)

    liquidity = (price_win * volume_win).mean()
    top_stocks = liquidity.nlargest(min(top_n, len(liquidity))).index

    return price_df[top_stocks], volume_df[top_stocks].astype(float)

# --- Fill missing values with average of 2 before and 2 after ---
def fill_na_with_2before_2after(df: pd.DataFrame, min_neighbors=1, require_all=False):
    s1 = df.shift(1)
    s2 = df.shift(2)
    s_1 = df.shift(-1)
    s_2 = df.shift(-2)

    neighbor_sum = s1.fillna(0) + s2.fillna(0) + s_1.fillna(0) + s_2.fillna(0)
    neighbor_count = (
        s1.notna().astype(int)
        + s2.notna().astype(int)
        + s_1.notna().astype(int)
        + s_2.notna().astype(int)
    )

    neighbor_mean = neighbor_sum / neighbor_count.where(neighbor_count > 0)

    if require_all:
        fill_values = neighbor_mean.where(neighbor_count == 4)
    else:
        fill_values = neighbor_mean.where(neighbor_count >= min_neighbors)

    return df.where(df.notna(), fill_values)

# --- Universe selection based on liquidity ---
def select_universe(prices_df: pd.DataFrame, volume_df: pd.DataFrame, date: pd.Timestamp, top_n=300, lookback_window=252):
    # filter stocks with price data on the given date
    index_date = prices_df.index.get_loc(date)
    list_stocks_notna_today = prices_df.columns[prices_df.loc[date].notna()]
    price = prices_df[list_stocks_notna_today].iloc[(index_date - lookback_window+1):(index_date + 1)]
    volume = volume_df[list_stocks_notna_today].iloc[(index_date - lookback_window+1):(index_date + 1)]

    # common columns
    common_columns = price.columns.intersection(volume.columns)
    price = price[common_columns]
    volume = volume[common_columns]

    # handle missing values by forward fill then drop na
    price = price.ffill(limit=2)
    price = price.dropna(axis=1, how='any')
    columns = price.columns
    volume = volume[columns]
    # fill volume missing values with average of previous and next valid values
    volume = fill_na_with_2before_2after(volume, min_neighbors=1, require_all=False)
    volume = volume.dropna(axis=1, how='any')
    columns = volume.columns
    price = price[columns]
    
    # return price and volume for the top N stocks by liquidity
    return top_liquidity_stocks(price, volume, top_n=top_n, average_window=lookback_window)

# --- Returns calculation ---
def compute_returns(price_df: pd.DataFrame, volume_df: pd.DataFrame=None, with_volume=False, volume_average_window=10):
    # basic returns calculation
    returns = price_df.pct_change().dropna()

    # volume adjustment
    if with_volume and volume_df is not None:
        volume_average = volume_df.shift(1).rolling(volume_average_window).mean().dropna() # volume average based on previous day
        volume_average = volume_average.reindex(returns.index) 
        delta_volume = volume_average.diff().dropna() 
        returns = returns*(volume_average/delta_volume)
    return returns.dropna()
# ----------------------------------------------------------------------------

# 2. PCA and factor modeling
# --- Standardize returns ---
def standardize_returns(returns_df: pd.DataFrame):
    mean_i = returns_df.mean(axis=0)
    std_i  = returns_df.std(axis=0, ddof=1).replace(0, np.nan)
    standardized = (returns_df - mean_i) / std_i
    return standardized.dropna(axis=1)

# --- Empirical correlation ---
def compute_empirical_correlation(standardized_returns: pd.DataFrame):
    M = standardized_returns.shape[0]
    return (standardized_returns.values.T @ standardized_returns.values) / (M - 1)

# --- Eigen decomposition sorted ---
def eigen_decomposition_sorted(corr_matrix: np.ndarray):
    eigvals, eigvecs = np.linalg.eigh(corr_matrix)
    idx = np.argsort(eigvals)[::-1]
    return eigvals[idx], eigvecs[:, idx]

# --- PCA engine ---
import numpy as np
import pandas as pd

def window_pca_engine(
    returns_df: pd.DataFrame,
    fixed_factors: bool = False,
    n_factors: int = 15,
    variance_cutoff: float = 0.55
):
    if returns_df.shape[1] < 2:
        raise ValueError("returns_df must contain at least 2 stocks")

    # độ lệch chuẩn từng cổ phiếu
    std_stocks = returns_df.std(ddof=1)

    # loại các cột có std = 0 hoặc NaN
    valid_cols = std_stocks[(std_stocks > 0) & std_stocks.notna()].index
    returns_df = returns_df[valid_cols]
    std_stocks = std_stocks[valid_cols]

    if returns_df.shape[1] < 2:
        raise ValueError("Not enough valid stocks after removing zero-variance columns")

    # PCA
    Z = standardize_returns(returns_df)
    C = compute_empirical_correlation(Z)
    eigvals, eigvecs = eigen_decomposition_sorted(C)

    # chọn số factor
    if fixed_factors:
        k = min(n_factors, returns_df.shape[1] - 1)
    else:
        cumvar = np.cumsum(eigvals) / eigvals.sum()
        k = int(np.searchsorted(cumvar, variance_cutoff)) + 1
        k = min(k, returns_df.shape[1] - 1)

    # eigenportfolios
    V = eigvecs[:, :k]
    Q = V / std_stocks.to_numpy().reshape(-1, 1)

    pca_dict = {
        "Q": Q,                           # quantity of stocks in each factor
        "V": V,                           # eigenvectors                                              
        "k": k,                           # number of factors
        "stocks": returns_df.columns,     # stock names
        "eigenvalues": eigvals[:k],       # eigenvalues of selected factors
    }

    return pca_dict
# ----------------------------------------------------------------------------

# 3. Model regression and residuals
# --- Compute PCA residuals ---
def compute_pca_residuals(returns_df: pd.DataFrame, pca_dict: dict, window=60):
    # take last 'window' returns for regression
    returns_win = returns_df.iloc[-window:]

    Q = pca_dict["Q"]
    stocks = pca_dict["stocks"]

    # check stocks match
    if list(stocks) != list(returns_df.columns):
        raise ValueError("PCA stocks do not match returns stocks")

    # matrix returns
    R = returns_win[stocks].values     # (window × N)

    # factor returns
    F = R @ Q                                    # (window × k)

    # design matrix with intercept
    X = np.column_stack([np.ones(window), F])     # (window × (k+1))

    # OLS solution
    XtX_inv = np.linalg.pinv(X.T @ X)
    B = XtX_inv @ (X.T @ R)                       # ((k+1) × N)

    # fitted returns
    fitted = X @ B                                # (window × N)

    # residuals
    residuals = R - fitted

    # convert to DataFrame
    residuals_df = pd.DataFrame(
        residuals,
        index=returns_win.index,
        columns=stocks
    )

    return residuals_df      

    
# --- OU fitting ---
# sửa lại thêm case không limit kappa 
def fit_ou_model(
    residuals,
    window=60,
    limit=False,
    kappa_min=8.4,
    min_obs=40,
    return_details=True
 ):
    kappa = pd.DataFrame(index=residuals.index, columns=residuals.columns, dtype=float)
    mu = kappa.copy()
    sigma_eq = kappa.copy()

    for col in residuals.columns:
        series = residuals[col]

        for t in range(window, len(series)):
            eps_win = series.iloc[t - window:t].dropna()

            if len(eps_win) < min_obs:
                continue

            Xcum = eps_win.cumsum().values
            if len(Xcum) < 3:
                continue

            x_prev = Xcum[:-1]
            x_next = Xcum[1:]

            A = np.column_stack([np.ones(len(x_prev)), x_prev])

            try:
                a, b = np.linalg.lstsq(A, x_next, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue

            resid_ar = x_next - (a + b * x_prev)
            var_xi = np.var(resid_ar, ddof=1)

            if not np.isfinite(b) or not np.isfinite(var_xi):
                continue
            if var_xi <= 1e-12:
                continue
            if b <= 0 or b >= 0.9672:
                continue

            kappa_val = -np.log(b) * 252.0
            mu_val = a / (1.0 - b)
            sigma_eq_val = np.sqrt(var_xi / (1.0 - b**2))

            if not np.isfinite(kappa_val) or not np.isfinite(mu_val) or not np.isfinite(sigma_eq_val):
                continue
            if sigma_eq_val <= 1e-12:
                continue
            if limit and kappa_val < kappa_min:
                continue

            date = series.index[t]
            kappa.loc[date, col] = kappa_val
            mu.loc[date, col] = mu_val
            sigma_eq.loc[date, col] = sigma_eq_val

    # Combined daily dataframe: rows=days, columns=(Stock, Stat)
    # Example columns: ('AAA', 'kappa'), ('AAA', 'mu'), ('AAA', 'sigma_eq')
    ou_daily_df = pd.concat({'kappa': kappa, 'mu': mu, 'sigma_eq': sigma_eq}, axis=1)
    ou_daily_df = ou_daily_df.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0)
    ou_daily_df.columns.names = ['Stock', 'Stat']

    if return_details:
        return ou_daily_df, kappa, mu, sigma_eq
    return ou_daily_df

# --- S-score ---
# thêm hàm vẽ phân phối 
def compute_s_score(residuals, mu, sigma_eq, window=60, min_obs=40):
    sscore_df = pd.DataFrame(index=residuals.index, columns=residuals.columns, dtype=float)

    for col in residuals.columns:
        eps = residuals[col]

        for t in range(window, len(eps)):
            eps_win = eps.iloc[t - window:t].dropna()
            if len(eps_win) < min_obs:
                continue

            x_t = eps_win.cumsum().iloc[-1]
            mu_t = mu.iloc[t, mu.columns.get_loc(col)]
            sigma_t = sigma_eq.iloc[t, sigma_eq.columns.get_loc(col)]

            if pd.isna(mu_t) or pd.isna(sigma_t) or sigma_t <= 1e-12:
                continue

            sscore_df.iloc[t, sscore_df.columns.get_loc(col)] = (x_t - mu_t) / sigma_t

    return sscore_df


# --- Signal generation ---
# sửa thành còn mỗi long
# sửa thêm điều kiện accept signal sau t+2.5 ngày  
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
# sửa leverage, thay thành equal weight 
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



# sửa hedge theo FU
def backtest_pca_strategy(weights, returns, fu_vn30_returns,
                           initial_equity=100, slippage=0.0005, beta_window=60):
    fu_vn30_returns = fu_vn30_returns.squeeze()
    common      = (weights.index
                   .intersection(returns.index)
                   .intersection(fu_vn30_returns.index))
    weights     = weights.loc[common].fillna(0)
    returns     = returns.loc[common].fillna(0)
    fu_vn30_returns = fu_vn30_returns.loc[common].fillna(0)
    stock_ret   = (weights * returns).sum(axis=1)
    rolling_cov = stock_ret.rolling(beta_window).cov(fu_vn30_returns)
    rolling_var = fu_vn30_returns.rolling(beta_window).var().replace(0, np.nan)
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
        pnl_spy   = Q_spy * fu_vn30_returns.iloc[t]
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
