import os

from googleapiclient.discovery import build


def google_search(search_term, api_key, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    return res.get("items", [])


# Usage example
results = google_search(
    "Python programming",
    api_key=os.environ.get("GOOGLE_API"),
    cse_id=os.environ.get("SEARCH_ID"),
    num=10,
)

for result in results:
    print(f"Title: {result['title']}")
    print(f"URL: {result['link']}")
    print(f"Description: {result['snippet']}\n")
