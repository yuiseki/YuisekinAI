"""Extract the text of every Japanese law from the e-Gov bulk XML archive.

The archive is at
https://laws.e-gov.go.jp/bulkdownload?file_section=1&only_xml_flag=true
and arrives as all_xml.zip: one directory per law, each holding one XML file.

Laws are outside copyright under Article 13(1) of the Copyright Act, and are
additionally published under the Government Standard Terms of Use 2.0.
"""

import argparse
import sys
import zipfile
from xml.etree.ElementTree import ParseError, fromstring


def iter_laws(zip_path):
    """Yield (member name, flattened text) for each law XML in the archive."""
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.endswith(".xml"):
                continue
            try:
                root = fromstring(z.read(name))
            except ParseError as exc:
                # Do not skip silently: a malformed law is a fact about the
                # archive, not something to paper over.
                print(f"PARSE FAILURE {name}: {exc}", file=sys.stderr)
                continue
            text = "\n".join(s.strip() for s in root.itertext() if s and s.strip())
            if text:
                yield name, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    laws = 0
    nbytes = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for _, text in iter_laws(args.zip_path):
            laws += 1
            nbytes += len(text.encode("utf-8"))
            out.write(text + "\n\n")
    print(f"laws\t{laws}")
    print(f"bytes\t{nbytes}")


if __name__ == "__main__":
    main()
