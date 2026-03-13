import datetime as dt


def get_since(N, timeframe='1d', from_time=None):
    """
    Computes a 'since' timestamp (ms) for fetch_ohlcv.

    Variation 1 (from_time=None): (N+1) periods back from now.
    Variation 2 (from_time set):  (N+1) periods back from that time.
        e.g. from_time=yesterday, timeframe='1d', N=100 → 100 days ago from yesterday.

    N: number of periods to go back.
    timeframe: '1d', '1h', '4h', etc.
    from_time: optional dt.datetime (or None for "now"). Naive treated as UTC.
    """
    if from_time is None:
        ref = dt.datetime.utcnow()
    else:
        ref = from_time
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=dt.timezone.utc)

    if timeframe == '1d':
        hours_per = 24
    elif timeframe.endswith('d'):
        hours_per = 24 * int(timeframe[:-1])
    elif timeframe.endswith('h'):
        hours_per = int(timeframe[:-1])
    elif timeframe.endswith('m'):
        hours_per = int(timeframe[:-1]) / 60
    else:
        raise ValueError("Unsupported timeframe, use e.g. '1d', '1h', '4h'")

    start = ref - dt.timedelta(hours=hours_per * (N + 1))
    start = start.replace(second=0, microsecond=0, minute=0)

    since = int(start.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    return since
