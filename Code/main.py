#main.py

from Code.config import ShellParams
from Code.runner import ShellRunner

if __name__ == "__main__":
    params = ShellParams(M = 5, b = 20, integrator = "rk4").validate() #esto llama a la clase shellParams, con M=5 y b=20, y luego valida los parámetros. Si son válidos, devuelve el objeto params.

    print(params.summary()) #imprime un resumen de los parámetros, incluyendo M, b, R_max, régimen, etc.

    results = ShellRunner(params).run() #Ahora pasa al runner
    print(results.summary())

    