SRC	:= $(wildcard main.tex *.tex)


KB.pdf: $(SRC) *.inc
	pdflatex egpgv19.tex
	bibtex egpgv19.aux
	pdflatex egpgv19.tex
	pdflatex egpgv19.tex


clean:
	rm -f *.glo *.lbl *.log *.dvi *.gls *.toc *.aux *.ist *.out *.glg *.pdf *.bbl *.blg *.lof *.brf

.PHONY: clean
