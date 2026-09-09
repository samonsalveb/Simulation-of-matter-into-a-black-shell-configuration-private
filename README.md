<<<<<<< HEAD
# Colapso gravitacional de una cáscara delgada → Black Shell

Tesis de maestría — Santiago Monsalve Bautista
Observatorio Astronómico Nacional (OAN), Universidad Nacional de Colombia.

Simulación del colapso de una cáscara de materia (polvo) usando las condiciones
de empalme de Israel, con Schwarzschild exterior y Minkowski interior, orientada
a caracterizar configuraciones de *black shell* cerca del horizonte.

## Estructura del repositorio

```
.
├── thesis/          # documento LaTeX
│   ├── main.tex
│   ├── references.bib
│   ├── chapters/    # un archivo por capítulo (con notas de alcance)
│   └── .latexmkrc
├── Code/            # paquete Python del simulador 
├── figures/         # figuras generadas por el código, usadas por la tesis
├── .gitignore
└── README.md
```

## Compilar la tesis

Requiere una distribución TeX. Recomendado **TinyTeX** (instala paquetes
faltantes solos al compilar, sin gestionar versiones):

```bash
# instalar TinyTeX (una sola vez)
# ver https://yihui.org/tinytex/  ->  instala tlmgr + latexmk

cd thesis
latexmk           # genera main.pdf
latexmk -pvc      # modo watch: recompila al guardar
latexmk -c        # limpia auxiliares (deja el PDF)
```

En VS Code: extensión **LaTeX Workshop** (usa el `.latexmkrc` automáticamente).
Activá SyncTeX para saltar entre el `.tex` y el PDF.

> **Formato UNAL:** este esqueleto es genérico para empezar hoy. La UNAL tiene
> plantilla oficial de tesis (biblioteca/SINAB); verificá el formato vigente con
> tu posgrado y migrá el `\documentclass` cuando corresponda. Los capítulos no
> cambian.

## Correr el código

```bash
cd code
python3 -m Code.main          # o tu script de entrada
python3 -m pytest             # suite de tests
```

Las figuras de la tesis se generan desde `code/` y se guardan en `figures/`.

## Flujo de versionado

```bash
git add -A
git commit -m "mensaje claro y específico"
git push
```

El **PDF compilado NO se versiona** (es artefacto de build; está en `.gitignore`).
Para distribuir el PDF final, adjuntalo a un *Release* de GitHub sobre un tag.

## Licencia

Falta elegir. Sugerencia: **MIT** (permisivo) para el código si querés que sea
reusable y citable, y considerá sacar un **DOI vía Zenodo** en el tag final para
poder citar el software en la tesis. Sin archivo `LICENSE`, legalmente el repo es
"todos los derechos reservados". Decisión tuya — agregá `LICENSE` cuando resuelvas.
