import numpy as np
import pandas as pd
import yfinance as yf


def download_prices(tickers, start, end):
    """Descarga precios de cierre ajustados (fechas x tickers)."""
    datos = yf.download(
        tickers, start=start, end=end,
        auto_adjust=True, progress=False,
    )
    return datos["Close"]


def clean_prices(precios, max_nan_frac=0.05):
    """Elimina activos con demasiados NaN y recorta a la ventana común."""
    frac_nan = precios.isna().mean()
    a_eliminar = frac_nan[frac_nan > max_nan_frac].index.tolist()
    precios_limpios = precios.drop(columns=a_eliminar)
    precios_limpios = precios_limpios.dropna()
    print("Eliminados:", a_eliminar)
    return precios_limpios


def compute_log_returns(precios):
    """Log-retornos diarios: ln(P_t / P_{t-1})."""
    log_ret = np.log(precios / precios.shift(1))
    return log_ret.dropna()