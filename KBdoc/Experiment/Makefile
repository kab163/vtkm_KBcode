SRC=main.tex
BIB=main.aux

main.pdf: 
	latexmk -pdf $(SRC)
	biblatex $(BIB)
	latexmk -pdf $(SRC)

clean:	
	rm *.pdf *.aux *.log *.fls

