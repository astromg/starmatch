#!/usr/bin/env python3
"""Run StarMatch and show matched stars on both FITS images in FitsView.

Usage:  python starmatch_view.py reference.ap field.ap [pixscale]

FITS images are expected next to the catalogues (``reference.fits``, ``field.fits``).
Requires:  pip install "starmatch[gui]"
"""

import sys
from pathlib import Path

from fitsview.FitsView_gui import FitsView
from PyQt5.QtWidgets import QApplication

from starmatch import match


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    ref_file, field_file = sys.argv[1], sys.argv[2]
    pixscale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

    r = match(ref_file, field_file, pixscale=pixscale)
    print(r)

    # matched field stars, and the same stars transformed into the reference frame
    field_x, field_y = r.field_xy.T
    ref_x, ref_y = r.transform.field_to_ref(field_x, field_y)

    app = QApplication(sys.argv)
    windows = []
    for cat, xs, ys in ((ref_file, ref_x, ref_y), (field_file, field_x, field_y)):
        fv = FitsView([])
        fv.fname = str(Path(cat).with_suffix(".fits"))
        fv.newFits()
        fv.ext_x = [xs]
        fv.ext_y = [ys]
        fv.show()
        fv.update()
        windows.append(fv)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
