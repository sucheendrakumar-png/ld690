"""Cache all posts, tags and categories to data/. Re-run any time; overwrites."""
import json, os, time, urllib.request, urllib.error
BASE = "https://www.lyricaldelights.com/wp-json/wp/v2"
os.makedirs("data", exist_ok=True)
def fetch_all(kind, fields=None):
    out, page = [], 1
    while True:
        url = f"{BASE}/{kind}?per_page=100&page={page}" + (f"&_fields={fields}" if fields else "")
        try:
            time.sleep(1)  # be gentle with the site
            batch = json.load(urllib.request.urlopen(url))
        except urllib.error.HTTPError as e:
            if e.code == 400: break  # past the last page
            raise
        if not batch: break
        out += batch; page += 1
    json.dump(out, open(f"data/{kind}.json", "w"), ensure_ascii=False)
    print(kind, len(out))
fetch_all("posts", "id,slug,link,date,modified,title,content,categories,tags,status")
fetch_all("tags", "id,name,slug,count")
fetch_all("categories", "id,name,slug,count")
