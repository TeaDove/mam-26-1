.DEFAULT_GOAL := pdf
export PATH := /Library/TeX/texbin:$(PATH)

.PHONY: docs docx pdf install-macos install-ubuntu

docs: pdf docx

docx:
	pandoc --from=latex --to=docx --reference-doc=docs/reference.docx --metadata=lang:ru-RU --fail-if-warnings --output=docs/research-passport.docx docs/research-passport.tex

pdf:
	mkdir -p docs/build
	cd docs && latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build research-passport.tex
	cp docs/build/research-passport.pdf docs/research-passport.pdf

install-macos:
	brew install --cask mactex-no-gui font-liberation
	brew install pandoc

install-ubuntu:
	sudo apt-get update
	sudo apt-get install -y make latexmk texlive-xetex texlive-latex-extra texlive-lang-cyrillic fonts-liberation pandoc
