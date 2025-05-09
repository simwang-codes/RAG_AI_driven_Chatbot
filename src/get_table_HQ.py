# This script defines functions for: scraping and cleaning the "video games considered the best" HTML table,
# following the hyperlinks of each game's publisher to scrape the country where the publisher is located,
# formalizing the country information,
# and storing everything in a clean, structured CSV table for visualization processes

import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm


# Originally, I tried to use SerpApi to retrieve each company's HQ location
# However, I realized I could access this information simply by following the hyperlink for each company in the original table
# Therefore, I defined the function below and used multiple if statements to ensure the retrieved HQ locations are standardized into country names
def get_best_games_table(save_path="best_video_games.csv"):
    # The function below retrieves the "Video Games Considered the Best" HTML table contents,
    # cleans the data, scrapes each video game company's HQ location in country format, and produces a CSV table for visualization.

    def get_HQ_from_wiki(wiki_url):
        try:
            resp = requests.get(wiki_url, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            
            infobox = soup.find("table", class_="infobox")
            if not infobox:
                return "Unknown"
            
            rows = infobox.find_all("tr")
            for row in rows:
                header = row.find("th")
                if header and "headquarters" in header.get_text(strip=True).lower():
                    cell = row.find("td")
                    if cell:
                        return cell.get_text(separator=" ", strip=True)
            return "Unknown"
        except Exception:
            return "Unknown"

    def formalize_country(hq_location):
        if not isinstance(hq_location, str):
            return "Unknown"

        hq_location = hq_location.lower()

        if re.search(r"\b(united states|u\.s\.|us)\b", hq_location) or \
           any(x in hq_location for x in ["california", "new york", "washington", "seattle"]):
            return "United States"
        if "japan" in hq_location:
            return "Japan"
        if "canada" in hq_location:
            return "Canada"
        if any(x in hq_location for x in ["united kingdom", "england", "london", "scotland"]):
            return "United Kingdom"
        if "france" in hq_location:
            return "France"
        if "germany" in hq_location:
            return "Germany"
        if "russia" in hq_location or "moscow" in hq_location:
            return "Russia"
        if "china" in hq_location or "shanghai" in hq_location or "beijing" in hq_location or \
           "hong kong" in hq_location or "taiwan" in hq_location or "taipei" in hq_location:
            return "China"
        if "south korea" in hq_location or "seoul" in hq_location:
            return "South Korea"
        if "australia" in hq_location or "sydney" in hq_location:
            return "Australia"
        if "italy" in hq_location or "rome" in hq_location:
            return "Italy"
        if "netherlands" in hq_location or "amsterdam" in hq_location:
            return "Netherlands"
        if "sweden" in hq_location or "stockholm" in hq_location:
            return "Sweden"
        if "poland" in hq_location or "warsaw" in hq_location:
            return "Poland"
        if "finland" in hq_location or "helsinki" in hq_location:
            return "Finland"
        if "spain" in hq_location or "madrid" in hq_location:
            return "Spain"
        if "brazil" in hq_location or "são paulo" in hq_location:
            return "Brazil"
        if "singapore" in hq_location:
            return "Singapore"

        return "Other"

    # In my first version, I mistakenly scraped the first <tbody>, which was not the one I wanted, so I fixed it here
    base_url = "https://en.wikipedia.org"
    main_url = "https://en.wikipedia.org/wiki/List_of_video_games_considered_the_best"
    resp = requests.get(main_url)
    soup = BeautifulSoup(resp.text, "html.parser")

    tbody = soup.find_all("tbody")[1]
    rows = tbody.find_all("tr")

    # Below, I used regex to remove any citation blobs from the headers.
    raw_hdrs = [th.get_text(strip=True) for th in rows[0].find_all("th")]
    hdrs = [re.sub(r"\(.*?\)\[.*?\]", "", h).strip() for h in raw_hdrs]
    hdrs = [h for h in hdrs if h.lower() != "ref."]

    data = []
    publisher_names = []
    publisher_links = []
    current_year = None
    for tr in rows[1:]:
        year_cell = tr.find("th")
        if year_cell:
            current_year = year_cell.get_text(strip=True)
        tds = tr.find_all("td")
        if not tds:
            continue

        # Here, I accessed each publisher's hyperlink and extracted their headquarters location from their Wikipedia page
        publisher_td = tds[2]  # 3rd column is Publisher
        publisher_name = publisher_td.get_text(strip=True)
        publisher_link = publisher_td.find("a")
        if publisher_link:
            link = base_url + publisher_link.get("href")
        else:
            link = None
        
        values = [td.get_text(strip=True) for td in tds[:-1]]
        data.append([current_year] + values)
        publisher_names.append(publisher_name)
        publisher_links.append(link)

    df = pd.DataFrame(data, columns=hdrs)

    # I switched the position of "Publisher" and "Original platform" here
    cols = df.columns.tolist()
    cols.remove("Original platform")
    pub_index = cols.index("Publisher")
    cols.insert(pub_index, "Original platform")
    df = df[cols]

# In my first version, I found that my program was doing repetitive scraping for companies whose HQ locations had already been retrieved,
# so now, when looking for a company's HQ location, if it has been previously scraped, the program directly inserts the existing result instead of scraping again
# This step significantly sped up the program, increasing the scraping speed by about 50%
    hq_cache = {}
    hq_locations = []

    for name, link in tqdm(zip(publisher_names, publisher_links), total=len(publisher_names), desc="Scraping HQ addresses"):
        if name in hq_cache:
            country = hq_cache[name]
        else:
            if link:
                hq_address = get_HQ_from_wiki(link)
            else:
                hq_address = "Unknown"
            country = formalize_country(hq_address)
            hq_cache[name] = country
        hq_locations.append(country)

    df["HQ Location"] = hq_locations

    df.to_csv(save_path, index=False)
    print(f"Saved {len(df)} rows to {save_path}")

