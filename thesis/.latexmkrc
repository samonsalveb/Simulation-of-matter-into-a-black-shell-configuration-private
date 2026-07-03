# .latexmkrc — configuración de compilación
# Uso:  cd thesis/  &&  latexmk        (compila main.tex)
#       latexmk -pvc                    (modo watch: recompila al guardar)
#       latexmk -c                      (limpia auxiliares, deja el PDF)

$pdf_mode   = 1;   # 1 = pdflatex. Cambiá a 4 si migrás a la plantilla UNAL
                   # con fuentes institucionales (Ancizar Sans -> requiere lualatex).
$bibtex_use = 2;   # corre bibtex y limpia el .bbl al hacer -c

@default_files = ('main.tex');

# Opcional: sacar los auxiliares del árbol (descomentá si te molesta el ruido).
# Puede complicar el visor/SyncTeX en algunos setups, por eso va apagado por defecto.
# $aux_dir = 'build';
# $out_dir = 'build';
