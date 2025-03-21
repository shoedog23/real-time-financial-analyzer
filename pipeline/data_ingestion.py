from sec_edgar_downloader import Downloader
import requests
from bs4 import BeautifulSoup

# Download historical 10-K filings for FAANG companies
def download_historical_filings():
    dl = Downloader("Personal", "natesh2310@gmail.com")
    companies = ["AAPL", "AMZN", "GOOG", "META", "NFLX"]
    for company in companies:
        dl.get("10-K", company, after="2018-01-01")

# Fetch new filings daily from SEC RSS feed
def fetch_new_filings():
    url = "https://www.sec.gov/Archives/edgar/usgaap.rss.xml"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "xml")
    
    filings = []
    for item in soup.find_all("item"):
        title = item.title.text
        link = item.link.text
        if "10-K" in title:
            filings.append(link)
    
    return filings

# Example usage:
if __name__ == "__main__":
    download_historical_filings()
    #new_filings = fetch_new_filings()
    #print("New Filings:", new_filings)