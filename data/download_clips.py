"""Download the three 20-second UFC clips used in the essay (Table 1).

Based on Figure 9 of the essay (partially generated with Grok, as declared there).
Changes for publishing:
  * clips are saved as videos/clip1.mp4, clip2.mp4, clip3.mp4 so they match the
    numbering in Table 1 and the scripts
  * trimming uses yt-dlp's `download_ranges` option, which is the one the Python API reads
  * the YouTube tracking parameter (?si=...) is removed from the URL

Requires: pip install yt-dlp   (and FFmpeg on your PATH, for trimming/remuxing)
The footage belongs to the UFC and is NOT included in this repository.
"""
from yt_dlp import YoutubeDL
from yt_dlp.utils import download_range_func

URL = "https://youtu.be/Pzi8UFEWRxI"  # "Jon Jones vs Lyoto Machida | FULL FIGHT | UFC Classics"

# (clip number, start, end, description)
CLIPS = [
    (1, "00:00:40", "00:01:00", "Baseline movements"),
    (2, "00:02:15", "00:02:35", "Active strikes"),
    (3, "00:01:00", "00:01:20", "Complex striking sequence"),
]


def to_seconds(timestamp):
    hours, minutes, seconds = (int(part) for part in timestamp.split(":"))
    return hours * 3600 + minutes * 60 + seconds


BASE_OPTS = {
    "format": "bestvideo[height<=720][fps=30]",  # 1280x720 at 30 FPS
    "postprocessors": [{"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"}],  # ensure MP4 output
    "force_keyframes_at_cuts": True,  # cut exactly at the requested times
}

for number, start, end, description in CLIPS:
    print(f"Downloading clip {number}: {description} ({start}-{end})")
    opts = BASE_OPTS.copy()  # copy to avoid overwriting
    opts["outtmpl"] = f"videos/clip{number}.%(ext)s"
    opts["download_ranges"] = download_range_func(None, [(to_seconds(start), to_seconds(end))])  # trim during download
    with YoutubeDL(opts) as ydl:
        ydl.download([URL])
