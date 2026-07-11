#!/usr/bin/env python3
"""JS-rendered re-check of ambiguous PRESAGE KB links.

verify_kb_quotes.py uses plain `requests`, so JavaScript-rendered pages
(cropprotectionnetwork.org, canr.msu.edu, ncbi/pmc, ...) return only nav chrome
and their quotes get falsely flagged quote_not_found / url_blocked. This pass
re-fetches every ambiguous URL with a real headless Chromium (Playwright),
extracts the rendered visible text, and re-matches the stored quotes so the
verdicts reflect what a browser actually sees.

Consumes/updates artifacts/quote_check.json in place (adds `rendered` + updated
quote_status/ratio; url status -> alive_rendered / render_dead / render_error).
Rendered page text cached in artifacts/page_cache/render_<sha1>.txt.

Usage:
    python scripts/render_recheck.py                 # render all ambiguous urls
    python scripts/render_recheck.py --limit 40      # smoke test
    python scripts/render_recheck.py --conc 8        # concurrency (default 6)
"""
import json, os, sys, asyncio, hashlib, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_kb_quotes import norm, match_quote, PAGE_DIR, CACHE_JSON, CSVOUT, write_csv, summarize
from playwright.async_api import async_playwright

AMBIG = {"quote_not_found", "quote_partial", "url_blocked",
         "url_unfetched_text", "url_unknown"}
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def render_cache(url):
    return os.path.join(PAGE_DIR, "render_" + hashlib.sha1(url.encode()).hexdigest() + ".txt")


def urls_to_render(results):
    out = []
    for u, d in results.items():
        if u.lower().endswith(".pdf"):
            continue  # PDFs handled separately, not renderable as text
        if d["status"] == "dead":
            continue
        if any(q["quote_status"] in AMBIG for q in d["quotes"]):
            out.append(u)
    return out


async def render_one(context, url, sem, use_cache):
    cp = render_cache(url)
    if use_cache and os.path.exists(cp):
        return url, "cache", open(cp, encoding="utf-8", errors="ignore").read()
    async with sem:
        page = await context.new_page()
        try:
            resp = await page.goto(url, wait_until="domcontentloaded", timeout=35000)
            await page.wait_for_timeout(2800)
            text = await page.inner_text("body")
            os.makedirs(PAGE_DIR, exist_ok=True)
            open(cp, "w", encoding="utf-8").write(text)
            code = resp.status if resp else 0
            status = "ok" if (resp and resp.status < 400) else f"http{code}"
            return url, status, text
        except Exception as e:
            return url, "error:" + type(e).__name__, ""
        finally:
            await page.close()


async def main_async(args):
    results = json.load(open(CACHE_JSON))
    urls = urls_to_render(results)
    if args.limit:
        urls = urls[:args.limit]
    print(f"rendering {len(urls)} ambiguous URLs with headless Chromium "
          f"(conc={args.conc})")

    sem = asyncio.Semaphore(args.conc)
    done = 0
    t0 = time.time()
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        context = await browser.new_context(user_agent=UA,
                                             viewport={"width": 1280, "height": 900})

        async def work(u):
            nonlocal done
            r = await render_one(context, u, sem, args.use_cache)
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(urls)}  ({time.time()-t0:.0f}s)")
            return r

        rendered = await asyncio.gather(*[work(u) for u in urls])
        await browser.close()

    # re-match
    for url, rstatus, text in rendered:
        d = results[url]
        pnorm = norm(text) if text else ""
        if pnorm:
            d["status"] = "alive_rendered"
            d["text_len"] = len(text)
        elif rstatus.startswith("error") or rstatus.startswith("http"):
            d["status"] = "render_error"
        for q in d["quotes"]:
            if q["quote_status"] not in AMBIG:
                continue  # already found via requests; leave it
            if pnorm:
                qs, ratio, found, total = match_quote(q["quote"], pnorm)
                q["quote_status"] = qs
                q["ratio"] = round(ratio, 3)
                q["frags_found"] = found
                q["frags_total"] = total
            q["rendered"] = True

    json.dump(results, open(CACHE_JSON, "w"), indent=1)
    write_csv(results)
    print("\n=== AFTER RENDER RE-CHECK ===")
    summarize(results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conc", type=int, default=6)
    ap.add_argument("--use-cache", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH",
                          "/work/mech-ai-scratch/tirtho/.pw_browsers")
    asyncio.run(main_async(args))
