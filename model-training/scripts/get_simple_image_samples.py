#!/usr/bin/env python3
"""Download sample images for a list of search phrases.

Edit the keywords list in this file, then run the script. Each phrase
is searched on Google Images, and up to 10000 pictures are saved for it.

    python get_simple_image_samples.py

Requires the simple-image-download package and a network connection.

Keywords
Put one search phrase in each list entry:

    keywords = [
        "person holding a rifle",
        "handgun on a table",
    ]

A comma inside one entry is split into separate searches. An empty
list downloads nothing.

Where files are saved
Pictures are written under simple_images/ in the current working
directory. Each phrase gets its own folder, named with that phrase:

    simple_images/
      handgun on a table/
        handgun on a table_1.jpg
        handgun on a table_2.jpg

Every saved file name ends in .jpg. The bytes are the file returned by
the source address. The search keeps results whose address contains
jpg, jpeg, png, gif, or ico.

The script prints each phrase before it starts that download. If the
Google Images page cannot be reached, the script stops.

Examples
From the folder where simple_images/ should be created:

    python get_simple_image_samples.py
"""

import argparse

from simple_image_download import simple_image_download as simp

response = simp.simple_image_download

# One Google Images search per entry. Each entry is saved in its own
# folder under simple_images/ in the current working directory.
keywords = [
]

# How many images to save for each keyword.
IMAGES_PER_KEYWORD = 10000


def main() -> None:
    # Show the guide before the flags, and the examples after them.
    overview, _marker, examples = __doc__.partition("Examples\n")

    parser = argparse.ArgumentParser(
        description=overview.strip(),
        epilog="Examples\n" + examples.rstrip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.parse_args()

    for kw in keywords:
        print(f"Downloading images for: {kw}")
        response().download(kw, IMAGES_PER_KEYWORD)


if __name__ == "__main__":
    main()
