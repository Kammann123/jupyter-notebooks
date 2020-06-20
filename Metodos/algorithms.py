from numpy import *

def euler(x0, f, t0, tf, n):
    """ Resolución de un sistema o una ecuación diferencial ordinaria a partir del método Euler, de 1° orden.
        :parametro x0: Condiciones iniciales del problema
        :parametro f: Función de carga
        :parametro t0: Tiempo inicial
        :parametro tf: Tiempo final
        :parametro n: Cantidad de puntos
    """
    t = [t0]
    x = [x0]
    h = (tf - t0) / n
    for i in range(1, n+1):
        t.append(t[i - 1] + h)
        x.append(x[i - 1] + h * f(t[i - 1], x[i - 1]))
    return array(t), array(x)

def heunn(x0, f, t0, tf, n):
    """ Calcula la evolución del sistema o ecuación diferencial empleando como método de aproximación
        el método de Heunn, un método Runge Kutta de segundo orden que emplea aproximación por el área del trapecio.
        :parametro x0: Estado inicial, condiciones iniciales
        :parametro f: Función de carga f(t,x)
        :parametro t0: Tiempo inicial
        :parametro tf: Tiempo final
        :parametro n: Cantidad de puntos
    """
    h = (tf - t0) / n
    t = [t0]
    x = [x0]
    for i in range(1, n+1):
        t.append(t[i - 1] + h)
        k1 = f(t[i-1], x[i-1])
        k2 = f(t[i-1] + h, x[i - 1] + h * k1)
        x.append(x[i - 1] + 0.5 * h * (k1 + k2))
    return array(t), array(x)

def jacobi(A: array, b: array, x0: array, iter_count: int):
    """ Resuelve el sistema lineal mediante el enfoque iterativo
        del método de Jacobi. Para lo cual se requieren matrices que sean
        estrictamente diagonales dominantes. Resuelve:
                                A * x = b
        @param A: Matriz del sistema
        @param b: Condiciones del problema
        @param x0: Condiciones iniciales del problema
        @param iter_count: Cantidad de iteraciones
    """
    if type(A) is not array:
        A = array(A)
    if type(b) is not array:
        b = array(b)
    if type(x0) is not array:
        x0 = array(x0)
    x = zeros((iter_count + 1, len(x0)))
    x[0] = x0
    for n in range(1, iter_count + 1):
        for i in range(len(b)):
            x_i = b[i]
            for j in range(len(b)):
                if j != i:
                    x_i -= A[i][j] * x[n-1][j]
            x_i = x_i / A[i][i]
            x[n][i] = x_i
    return x

def gauss_seidel(A: array, b: array, x0: array, iter_count: int):
    """ Resuelve el sistema lineal mediante el enfoque iterativo
        del método de Gauss Seidel. Para lo cual se requieren matrices que sean
        estrictamente diagonales dominantes o bien definida positiva. Resuelve:
                                A * x = b
        @param A: Matriz del sistema
        @param b: Condiciones del problema
        @param x0: Condiciones iniciales del problema
        @param iter_count: Cantidad de iteraciones
    """
    if type(A) is not array:
        A = array(A)
    if type(b) is not array:
        b = array(b)
    if type(x0) is not array:
        x0 = array(x0)
    x = zeros((iter_count + 1, len(x0)))
    x[0] = x0
    for n in range(1, iter_count + 1):
        for i in range(len(b)):
            x_i = b[i]
            for j in range(len(b)):
                if j < i:
                    x_i -= A[i][j] * x[n][j]
                elif j > i:
                    x_i -= A[i][j] * x[n-1][j]
            x_i = x_i / A[i][i]
            x[n][i] = x_i
    return x

def biseccion(f: callable, a: float, b: float, iter_count: int):
    """ Resuelve el problema de la ecuación no lineal en el intervalo dado,
        básicamente busca el resultado a la ecuación,
            f(x) = 0 con x en [a, b] realizando iter_count iteraciones.
        @param f: Función que describe el problema
        @param a: Cota inferior del intervalo
        @param b: Cota superior del intervalo
        @param iter_count: Cantidad de iteraciones
    """
    if f(a) * f(b) < 0:
        for i in range(1, iter_count + 1):
            c = (a + b) / 2
            if f(a) * f(c) > 0:
                a = c
            elif f(a) * f(c) < 0:
                b = c
            else:
                break
        return c
    else:
        raise ValueError("No cumple las condiciones del problema, puede no haber un cero.")

def punto_fijo(g: callable, x0: float, iter_count: int):
    """ Resolver ecuaciones no lineales por punto fijo.
        @param g: Función de la ecuación a resolver g(x) = x
        @param x0: Condicion inicial
        @param iter_count: Contador de iteraciones
    """
    x = [x0]
    for i in range(1, 1 + iter_count):
        x.append(g(x[i - 1]))
    return x

def newton_raphson(f: callable, f_: callable, x0: float, iter_count: int):
    """ Resolver ecuaciones no lineales por metodo de Newton Raphson
        @param f: Función del problema
        @param f_: Derivada de la función
        @param x0: Condicion inicial
        @param iter_count: Cantidad de pasos
    """
    x = [x0]
    for i in range(1, 1 + iter_count):
        x_i = x[i - 1] - f(x[i - 1])/f_(x[i - 1])
        x.append(x_i)
    return x

def secante(f: callable, x0: float, iter_count: int):
    """ Resolver ecuaciones no lineales por metodo de Secante
        @param f: Función del problema
        @param x0: Condicion inicial
        @param iter_count: Cantidad de pasos
    """
    x = [x0]
    for i in range(1, 1 + iter_count):
        x_i = x[i - 1] - f(x[i - 1]) * ((x[i - 1] - x[i - 2]) / (f(x[i - 1]) - f(x[i - 2])))
        x.append(x_i)
    return x

def lagrange_interpolation(a, b, n, f, cheby):
    """
        Encuentra el polinomio interpolador de 'f' en [a, b] usando polinomios de lagrange de orden n
        @param cheby: True si se usan nodos de chebyshev, de lo contrario equiespaciados
        @return (polinomio, (nodos_x, f(nodos_x)))
            Los coeficientes del polinomio son: [an,..., a1, a0] que el orden en que np.polyval() recibe los coeficientes 
            
        ¡Esta función fue realizada por Rafael Nicolás Trozzo, yo simplemente la copie y pegué.!
    """
    if cheby:
        zj = [np.cos((2*j+1)/(2*(n+1))*np.pi) for j in range(n+1)]
        x_nodes = [(b+a)/2 + (b-a)/2*z for z in zj] 
    else:
        x_nodes = np.linspace(a, b, n + 1)
    y_nodes = [f(x) for x in x_nodes]
    # hay n+1 polinomios con n+1 coeficientes
    lag_pols = np.zeros((n + 1, n + 1)) 
    for k in range(len(x_nodes)):
        # Formamos el polinomio de lagrange k a partir de sus raices
        lag_roots = np.delete(x_nodes, k)
        lag_pols[k] = np.flip(poly.polyfromroots(lag_roots)) / np.prod(x_nodes[k] - lag_roots) * y_nodes[k]
        print(lag_pols[k])  # comentar si no se quieren ver los polinomios de Lagrange
    interp_poly = np.sum(lag_pols, axis=0)
    
    return (interp_poly, (x_nodes, y_nodes))