import json, time
import yfinance as yf

"""
THIS CODE DOWNLOAD ESTIMATED PRICE OF ASSET
AVAILABLE ON YAHOO FINANCE
"""

# Constants defining
TICKERS = ['TSLA', 'KVUE', 'JNJ', 'AMC', 'NU', 'AMD', 'NVDA', 'F', 'BAC', 'AAPL', 'X', 'PLTR', 'DLO', 'AMZN', 'COHR', 'RIVN', 'PBR', 'SNAP', 'INTC', 'BBD', 'T', 'LCID', 'BABA', 'TGT', 'MARA', 'SE', 'PFE', 'DNA', 'CCL', 'JD', 'AAL', 'ITUB', 'SOFI', 'GOOGL', 'ET', 'CSCO', 'RIOT', 'HBAN', 'VALE', 'CLF', 'CMCSA', 'GRAB', 'META', 'IONQ', 'PLUG', 'MSFT', 'PYPL', 'TJX', 'UBER', 'SWN', 'VZ', 'PINS', 'GOOG', 'OPEN', 'GM', 'GOLD', 'FNMA', 'LYFT', 'JBLU', 'NOK', 'KEY', 'XOM', 'TSEM', 'C', 'XPEV', 'GGB', 'NYCB', 'MRVL', 'MPW', 'DIS', 'BRFS', 'AES', 'TFC', 'DKNG', 'ERIC', 'RIG', 'AGNC', 'MQ', 'KDP', 'WFC', 'BCS', 'MU', 'TME', 'COIN', 'CHPT', 'PCG', 'BMY', 'KMI', 'NCLH', 'ONON', 'LU', 'SIRI', 'FCX', 'DISH', 'KGC', 'PBR-A', 'ABEV', 'USB']

INFOS_TO_GET = [
    "symbol",
    "currency",
    "currentPrice",
    "targetHighPrice",
    "targetLowPrice",
    "targetMeanPrice",
    "targetMedianPrice",
    "recommendationKey",
    "numberOfAnalystOpinions",
]

SEC_TO_WAIT = 1

today = time.strftime("%Y-%m-%d")
print(f"Today is {today}")

# Function defining
def get_ticker_info(symbol):
    ticker = yf.Ticker(symbol)
    info = ticker.info
    dico_to_get = {}
    for info_to_get in INFOS_TO_GET:
        if info_to_get not in info.keys():
            return False, info_to_get
        dico_to_get[info_to_get] = info[info_to_get]
    return True, dico_to_get

# Execution
all_tickers_dico = {}
for symbol in TICKERS:
    print(symbol)
    ok, result = get_ticker_info(symbol)
    if ok:
        all_tickers_dico[symbol] = result
    else:
        print(f"{symbol} : doesn't have '{result}'")
    time.sleep(SEC_TO_WAIT)

with open(f"yahoo/{today}.json", "w") as outfile:
    json.dump(all_tickers_dico, outfile, indent=4)
