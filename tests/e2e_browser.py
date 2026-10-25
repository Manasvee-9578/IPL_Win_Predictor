"""Optional end-to-end browser checks for ChaseIQ (not collected by pytest).

Needs the app running (python run.py), Google Chrome and:  pip install playwright
Run:  python tests/e2e_browser.py
"""

import re
from playwright.sync_api import sync_playwright

SEL = "[data-testid^=stBaseButton-segmented_control]"
res = []


def check(name, ok, extra=""):
    res.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL ") + name, extra)


with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True)
    pg = b.new_page(viewport={"width": 1440, "height": 1000})
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto("http://127.0.0.1:8501"); pg.wait_for_timeout(6000)
    txt = lambda: pg.inner_text("body")
    nav = lambda name: (pg.locator(SEL).filter(has_text=name).first.click(), pg.wait_for_timeout(3500))

    # ---- branding / navigation
    t = txt()
    check("brand is ChaseIQ with tagline", "ChaseIQ" in t and "Inside Every Chase." in t and "IPL Win Predictor" not in t)
    names = [pg.locator(SEL).nth(i).inner_text().strip().upper() for i in range(5)]
    check("nav has the five sections, no Log", names == ["LIVE CRICKET", "SIMULATOR", "REPLAY", "ANALYTICS", "MODEL"], names)
    check("opens on Live Cricket", "Live cricket" in t)

    # ---- live cricket: honest states + real data + filters
    check("model-scope notice present", "never predicts them" in t and "IPL Second-Innings Chase Model" in t)
    check("invalid/missing key explained, with free-key steps", ("rejected" in t or "No CricketData.org API key" in t) and "cricketdata.org" in t)
    check("no fabricated live match", pg.locator(".mcard-live").count() == 0)
    n_all = pg.locator(".mcard").count()
    check("recent results are real cards with a source", n_all > 0 and "Source: Cricsheet.org" in t, f"({n_all} cards)")
    check("every card is labelled HISTORICAL", pg.locator(".mcard .tag-hist").count() == n_all)
    pill = lambda label: pg.locator(".st-key-filters button").filter(has_text=re.compile(f"^{label}$", re.I)).first
    pill("India Men").click(); pg.wait_for_timeout(3500)
    cards = pg.locator(".mcard")
    only_women = all("India Women" in cards.nth(i).inner_text() for i in range(cards.count()))
    check("deselect India Men -> only India Women cards", cards.count() > 0 and cards.count() < n_all and only_women, f"({cards.count()} of {n_all})")
    pill("India Men").click(); pg.wait_for_timeout(3000)
    pill("T20I").click(); pg.wait_for_timeout(3500)
    cards = pg.locator(".mcard")
    check("deselect T20I -> only ODI/Test cards remain", cards.count() > 0 and all(
        "T20I" not in cards.nth(i).locator(".tag").first.inner_text().upper() for i in range(cards.count())), f"({cards.count()})")
    pill("T20I").click(); pg.wait_for_timeout(2500)
    pg.locator(SEL).filter(has_text="Upcoming").first.click(); pg.wait_for_timeout(3000)
    check("upcoming tab shows an honest empty state (no invented fixtures)", "Nothing to show" in txt() and pg.locator(".mcard").count() == 0)

    # ---- simulator
    nav("Simulator")
    check("simulator is HYPOTHETICAL and shows the model name", pg.locator(".bug .tag-hypo").count() == 1 and "IPL Second-Innings Chase Model" in txt())
    check("no 'save to log' button anywhere", "Save this scenario" not in txt() and "Prediction log" not in txt())
    pg.locator(".st-key-ball_6 button").click(); pg.wait_for_timeout(2500)
    check("bowling a six works", "96/3" in txt())

    # ---- replay + model
    nav("Replay")
    check("replay is HISTORICAL", pg.locator(".bug .tag-hist").count() == 1)
    pg.locator(".st-key-play button").click(); pg.wait_for_timeout(4500)
    check("replay playback advances", re.search(r"(\d+)\.(\d)\s*overs", txt(), re.I) is not None and "0.0 overs" not in txt().lower())
    nav("Model")
    check("model page states its scope", "never applies it to ODI, T20I or Test" in txt())
    nav("Analytics")
    check("analytics loads", "The IPL in numbers" in txt())
    pg.locator(SEL).filter(has_text="Seasons").first.click(); pg.wait_for_timeout(3000)       # click the ACTIVE sub-tab
    pg.locator(SEL).filter(has_text="Analytics").first.click(); pg.wait_for_timeout(3000)     # ...and the ACTIVE main tab
    check("clicking an active tab is a no-op (no widget-state warning)", "default value but also" not in txt()
          and pg.locator("[data-testid=stAlert]").count() == 0 and "The IPL in numbers" in txt())
    probe = pg.evaluate(r"""() => { const bad = []; let coloured = 0;
        for (const el of document.querySelectorAll('body *')) {
          const cs = getComputedStyle(el);
          for (const prop of ['color', 'backgroundColor', 'borderTopColor']) {
            const m = cs[prop].match(/rgba?\((\d+), (\d+), (\d+)(?:, ([\d.]+))?\)/);
            if (!m || (m[4] !== undefined && +m[4] < .3)) continue;
            const r = +m[1], g = +m[2], b = +m[3], mx = Math.max(r, g, b), mn = Math.min(r, g, b);
            if (mx - mn < 40) continue;
            coloured++;
            if (g === mx && g - Math.max(r, b) > 25) bad.push(prop + ':' + cs[prop]);
          } }
        return {coloured, bad: bad.slice(0, 5)}; }""")
    check("no green computed colours on the page", probe["coloured"] > 20 and not probe["bad"],
          f"({probe['coloured']} coloured properties inspected, green: {probe['bad']})")
    check("no page exceptions", pg.locator("[data-testid=stException]").count() == 0 and not errs, str(errs[:1]))

    # ---- mobile: no sideways scrolling on the new pages
    m = b.new_page(viewport={"width": 390, "height": 900})
    m.goto("http://127.0.0.1:8501"); m.wait_for_timeout(6000)
    for view in ["Live Cricket", "Simulator"]:
        if view != "Live Cricket":
            m.locator(SEL).filter(has_text=view).first.click(); m.wait_for_timeout(4000)
        sw = m.evaluate("() => { const e=document.querySelector('[data-testid=stMain]')||document.documentElement; return [e.scrollWidth, e.clientWidth]; }")
        check(f"mobile: {view} has no horizontal overflow", sw[0] <= sw[1] + 2, sw)
    b.close()

print(f"\n{sum(o for _, o in res)}/{len(res)} passed")
