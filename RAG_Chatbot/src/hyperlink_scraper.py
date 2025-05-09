import re
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

BASE_URL = "https://en.wikipedia.org"
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_video_games_considered_the_best"
OUTPUT_FILE = "best_games_test.txt"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0.0.0 Safari/537.36"
    )
}
MAX_WORKERS = 10  # adjust to your bandwidth

# reuse a session for HTTP connection pooling
session = requests.Session()
session.headers.update(HEADERS)

def extract_game_links_and_genres(limit=400):
    resp = session.get(WIKI_URL, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    games = []
    for table in soup.find_all("table", class_="wikitable"):
        for row in table.find_all("tr")[1:]:
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            a = cells[0].find("a", href=True)
            if not (a and a["href"].startswith("/wiki/")):
                continue
            title = a.get_text(strip=True)
            link  = BASE_URL + a["href"]
            genre = cells[1].get_text(strip=True)
            games.append((title, genre, link))
            if len(games) >= limit:
                return games
    return games

def fetch_and_parse(entry):
    title, genre, url = entry
    try:
        r = session.get(url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")

        content = soup.select_one("#mw-content-text .mw-parser-output")
        if not content:
            return title, genre, ""

        # collect all paragraph text
        raw = "\n".join(
            p.get_text(" ", strip=True)
            for p in content.find_all("p")
            if p.get_text(strip=True)
        )

        # 1) convert non-breaking spaces to regular spaces
        text = raw.replace('\xa0', ' ')

        # 2) strip out [ 1 ], [23], etc.
        text = re.sub(r'\[\s*\d+\s*\]', '', text)

        # 3) remove stray spaces before commas/periods
        text = re.sub(r'\s+([.,])', r'\1', text)

        # 4) remove any injected warning lines
        text = text.replace("⚠️ No article text found.", "")

        # 5) drop blank lines
        lines = [ln for ln in text.splitlines() if ln.strip()]
        clean = "\n".join(lines)

        return title, genre, clean

    except Exception:
        return title, genre, ""

def save_to_txt(game_data, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
            for title, genre, article in tqdm(
                exe.map(fetch_and_parse, game_data),
                total=len(game_data),
                desc="Scraping games",
                unit="game"
            ):
                # keep title/dashline/genre headers
                dash = "-" * len(title)
                f.write(f"\n{title}\n{dash}\nGenre: {genre}\n\n")
                # write cleaned article body
                f.write(article + "\n" if article else "⚠️ No article text found.\n")

if __name__ == "__main__":
    games = extract_game_links_and_genres(limit=337)
    save_to_txt(games, OUTPUT_FILE)
    print(f"\n🎉 Done! Output saved to: {OUTPUT_FILE}")
