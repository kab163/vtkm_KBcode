SRC=main.tex
BIB=main.aux

main.pdf: 
	latexmk -pdf $(SRC)
	bibtex $(BIB)
	latexmk -pdf $(SRC)

clean:	
	rm *.pdf *.aux *.log *.fls *.bbl *.blg

