from urllib.request import urlopen
from urllib.parse import urlencode
from bs4 import BeautifulSoup
import re # for REGularEXpression
import time, json

"""
THIS CODE DOWNLOAD ESTIMATED PRICE OF ASSET
AVAILABLE ON CNN
"""

# Constants defining
TICKERS = ['TSLA', 'KVUE', 'JNJ', 'AMC', 'NU', 'AMD', 'NVDA', 'F', 'BAC', 'AAPL', 'X', 'PLTR', 'DLO', 'AMZN', 'COHR', 'RIVN', 'PBR', 'SNAP', 'INTC', 'BBD', 'T', 'LCID', 'BABA', 'TGT', 'MARA', 'SE', 'PFE', 'DNA', 'CCL', 'JD', 'AAL', 'ITUB', 'SOFI', 'GOOGL', 'ET', 'CSCO', 'RIOT', 'HBAN', 'VALE', 'CLF', 'CMCSA', 'GRAB', 'META', 'IONQ', 'PLUG', 'MSFT', 'PYPL', 'TJX', 'UBER', 'SWN', 'VZ', 'PINS', 'GOOG', 'OPEN', 'GM', 'GOLD', 'FNMA', 'LYFT', 'JBLU', 'NOK', 'KEY', 'XOM', 'TSEM', 'C', 'XPEV', 'GGB', 'NYCB', 'MRVL', 'MPW', 'DIS', 'BRFS', 'AES', 'TFC', 'DKNG', 'ERIC', 'RIG', 'AGNC', 'MQ', 'KDP', 'WFC', 'BCS', 'MU', 'TME', 'COIN', 'CHPT', 'PCG', 'BMY', 'KMI', 'NCLH', 'ONON', 'LU', 'SIRI', 'FCX', 'DISH', 'KGC', 'PBR-A', 'ABEV', 'USB']

SEC_TO_WAIT = 1

today = time.strftime("%Y-%m-%d")
print(f"Today is {today}")


# Functions defining
def find_number(text, pre="", suf="", is_int=False, sign=False):
    if is_int:
        pattern = "\d+"
        f = int
    else:
        pattern = "\d+.\d+"
        f = float
    if sign:
        pattern = "[+-]" + pattern
    pattern = f"{pre}{pattern}{suf}"
    matches = re.findall(pattern, text)
    number = matches[0] # Get the first one
    number = number[len(pre):len(number)-len(suf)]
    return f(number)

def get_ticker_info(symbol):
    URL = 'https://money.cnn.com/quote/forecast/forecast.html'
    data = {'symb': symbol}
    try:
        info_to_get = "searching"
        page = urlopen(URL + "?" + urlencode(data))
        soup = BeautifulSoup(page, 'html.parser')

        info_to_get = "parsing"
        wsod_forecasts = soup.find('div', attrs={'id': 'wsod_forecasts'})
        paragraph = wsod_forecasts.find('p')
        text = paragraph.text

        info_to_get = "analysts"
        analysts = find_number(text, suf=" analyst", is_int=True)
        info_to_get = "median"
        median = find_number(text, pre="a median target of ", suf=",")
        info_to_get = "high"
        high = find_number(text, pre="a high estimate of ", suf=" and")
        info_to_get = "low"
        low = find_number(text, pre="a low estimate of ", suf=".")
        # increase = find_number(text, pre="median estimate represents a ", suf="%", sign=True) # Inutile car calculable
        info_to_get = "current"
        current = find_number(text, pre="last price of ", suf=".")
    except:
        return False, info_to_get
    
    dico_to_get = {
        "symbol": symbol,
        "analysts" : analysts,
        "median": median,
        "high": high,
        "low": low,
        "current": current
    }
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

with open(f"cnn/{today}.json", "w") as outfile:
    json.dump(all_tickers_dico, outfile, indent=4)