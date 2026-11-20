# Research project passport

From the `thesis` directory, install the dependencies with `make install-macos` or `make install-ubuntu`,
then run `make docs`. It produces the PDF at `docs/research-passport.pdf` and the editable Word
version at `docs/research-passport.docx`. Use `make pdf` to build only the PDF.

On macOS the installation target requires Homebrew and Make. On Ubuntu it requires Make and sudo
access. On a fresh Ubuntu installation, bootstrap Make with `sudo apt-get install make`.

## Formatting reference

The sample follows sections 3.2–3.3 of the [MISIS Novotroitsk branch guidelines][guidelines] for page layout,
body text, and headings. It uses A4, 30 mm left and 15 mm right margins, 20 mm top and bottom margins,
12.5 mm first-line indentation, justified text, 14-point body text, and 1.5 line spacing.
XeLaTeX uses Times New Roman when installed and the metric-compatible TeX Gyre Termes otherwise.

The one-page passport uses a centered 16-point heading and leaves the first page unnumbered.
Check the requirements of the graduating department before submitting a thesis.

[guidelines]:
https://nf.misis.ru/download/gisen/%D0%98%D0%B7%D0%BC%D0%B0%D0%B9%D0%BB%D0%BE%D0%B2%D0%B0%2C_%D0%A2%D0%BE%D1%80%D1%88%D0%B8%D0%BD%D0%B0_%D0%BC%D0%B5%D1%82%D0%BE%D0%B4._%D1%83%D0%BA%D0%B0%D0%B7._%D0%BF%D0%BE_%D0%BE%D1%84%D0%BE%D1%80%D0%BC%D0%BB%D0%B5%D0%BD%D0%B8%D1%8E_%D0%92%D0%9A%D0%A0_2022.pdf
