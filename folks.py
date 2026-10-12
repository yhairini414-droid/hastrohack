import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import os
from scipy.stats import f as dist_f

# ============== CONFIGURACIÓN ==============

ESPECTROS = {
    "W1": "W1_espectros.csv",
    "W2": "W2_espectros (1).csv"
}
CATALOGO = "catalogo_doppler_para_programador.csv"
CATALOGO_LOCAL = "catalogo_local.csv"     # energía, intensidad, g y transición
PARTICION = "particion_Q300K.csv"         # opcional: columnas especie, Q_300K

C_KMS = 299792.458

V_MIN = -2.0                # Límite inferior de velocidad a explorar (km/s)
V_MAX = 10.0                # Límite superior de velocidad a explorar (km/s)
PASO_V = 0.5                # Paso de búsqueda gruesa (velocidad inicial)
PASO_REFINAMIENTO = 0.02    # Paso de refinamiento fino (velocidad inicial)

VENTANA_MHZ = 2.0           # Margen del recorte global (velocidad inicial)
SIGMA_MIN_MHZ = 0.03
SIGMA_MAX_MHZ = 3.0
SIGMA_DETECCION = 5.5       # SNR de pico mínimo (por canal) para declarar una línea detectada
SIGMA_EXPLORATORIO = 3.0    # Debajo de esto es ruido; entre 3 y 5 sigma la línea queda 'dudosa'
SNR_INTEGRADO_MIN = 3.0     # SNR mínimo del área integrada de la línea
P_VALOR_F_MAX = 0.005       # Prueba F: la mejora de chi2 debe tener p < 0.005
MIN_TRANSICIONES = 2        # Transiciones independientes para dar por presente la molécula
N_CONTROL = 30              # Pruebas de control con posiciones desplazadas (falsas alarmas)
EJECUTAR_PRUEBAS = True     # Pruebas internas (ruido plano y líneas inyectadas) al iniciar

# --- Ajuste local por línea ---
LOCAL_MHZ = 3.0             # Media ventana alrededor de cada línea
SIGMA_LOCAL_MIN_MHZ = 0.15  # Anchura mínima de una línea (aprox. 0.6 canal)
SIGMA_LOCAL_MAX_MHZ = 2.0
TOL_EXPLORA_KMS = 2.0       # Cuánto puede moverse cada grupo de líneas (1ª pasada)
TOL_FINAL_KMS = 0.5         # Cuánto puede moverse alrededor de la velocidad final (2ª pasada)
CANALES_COHERENCIA = 1.0    # Diferencia máxima de velocidad entre líneas: 1 canal (resolución)
GRADO_BASE = 2              # Fondo local: 1 = recta, 2 = parábola (absorbe curvaturas suaves)

# --- Mezclas entre moléculas ---
TOL_MEZCLA_CANALES = 1.5    # Dos moléculas reclaman la misma línea si caen a menos de esto
FRAC_DOMINANTE = 0.8        # Una línea mezclada cuenta como limpia solo si, según el modelo
                            # LTE, a esta molécula le toca al menos esta fracción del pico
MIN_LIMPIAS = 2             # Líneas sin mezcla necesarias para dar la molécula por presente

# --- Chequeo de intensidades (LTE ópticamente delgado) ---
T_LTE_MIN, T_LTE_MAX = 20.0, 600.0   # Rango de temperatura de excitación (K)
FRAC_LTE = 0.3              # Tolerancia relativa del modelo LTE
PULL_LTE = 3.0              # Una línea es inconsistente si se desvía más de esto
FRAC_LTE_MINIMA = 0.5       # El chequeo LTE solo se usa si al menos esta fracción de las
                            # líneas >= 3 sigma encaja; si no, queda 'no concluyente'

# --- Velocidad sistémica de la fuente ---
V_SISTEMICA = None          # None = estimarla de los datos; o un número fijo en km/s (p. ej. 2.55)
TOL_V_SISTEMICA_KMS = 0.6   # Una línea detectada debe estar a menos de esto de la sistémica (~3 canales)
MIN_LINEAS_ANCLA = 3        # Líneas detectadas que necesita una molécula para servir de ancla
INC_MAX_MHZ = 0.3           # Líneas del catálogo con incertidumbre mayor no cuentan como evidencia

# --- Nivel 'tentativa': líneas débiles pero consistentes, para que el investigador las revise ---
SIGMA_TENTATIVA = 3.5       # SNR de pico mínimo de una línea tentativa
P_VALOR_F_TENT = 0.05       # Prueba F para tentativas
TOL_V_TENT_KMS = 1.0        # Tolerancia respecto a la velocidad sistémica (las débiles se miden peor)
TOL_ANCHO = 0.6             # Ancho de una tentativa (km/s) vs. el de las líneas ancla: +-60 %
PULL_LTE_TENT = 5.0         # Las tentativas toleran una desviación LTE mayor (opacidad, no-LTE)
MIN_TENTATIVAS = 2          # Tentativas independientes para marcar una molécula como 'tentativa'
RADIO_CATALOGO_MHZ = 1.0    # Control de falsas alarmas: un acierto a menos de esto de una línea del
                            # catálogo se considera 'explicado por el catálogo', no ruido

K_CM = 1.438777             # hc/k: kelvin por cm^-1
GRAFICA_SOLO_DETECTADAS = True   # True: SNR solo de líneas detectadas; False: todas
MAX_PANELES_ZOOM = 30


# ============== FUNCIONES DE PROCESO ==============

def estimar_ruido(y):
    """Ruido a partir de diferencias entre canales vecinos (MAD)."""
    diferencias = np.diff(np.asarray(y, dtype=float))
    mad = np.median(np.abs(diferencias - np.median(diferencias)))
    return 1.4826 * mad / np.sqrt(2)


def estimar_ruido_correlacionado(y):
    """Ruido por canal cuando los canales vecinos están correlacionados (espectros suavizados).
    La diferencia entre vecinos da sigma*sqrt(1-phi), así que se corrige con phi.
    Devuelve (sigma, phi), con phi = autocorrelación entre canales vecinos."""
    y = np.asarray(y, dtype=float)
    sigma_dif = estimar_ruido(y)
    y0 = y - np.median(y)
    s = np.std(y0)
    for _ in range(8):
        libre = np.abs(y0) < 3 * s
        s = np.std(y0[libre])
    ok = libre[:-1] & libre[1:]
    phi = float(np.clip(np.corrcoef(y0[:-1][ok], y0[1:][ok])[0, 1], 0.0, 0.95))
    return sigma_dif / np.sqrt(1.0 - phi), phi


def ruido_ar1(n, sigma, phi, rng):
    """Ruido sintético con correlación phi entre canales vecinos (para las pruebas)."""
    e = rng.normal(0, sigma * np.sqrt(1 - phi**2), n)
    y = np.empty(n)
    y[0] = rng.normal(0, sigma)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + e[t]
    return y


def modelo_lineas(x, parametros, frecuencias):
    """Continuo lineal más gaussianas de emisión (anchura común)."""
    n = len(frecuencias)
    base, pendiente = parametros[:2]
    amplitudes = parametros[2:2 + n]
    sigma = parametros[2 + n]

    modelo = base + pendiente * (x - np.mean(frecuencias))
    for amp, centro in zip(amplitudes, frecuencias):
        modelo = modelo + amp * np.exp(-0.5 * ((x - centro) / sigma)**2)
    return modelo


def ajustar_modelo(x, y, ruido, frecuencias):
    """Ajuste global (usado solo para la velocidad inicial)."""
    n = len(frecuencias)
    centro_ref = np.mean(frecuencias)
    base_inicial = np.median(y)
    amp_inicial = max(np.max(y) - base_inicial, ruido) / n

    inicial = [base_inicial, 0.0] + [amp_inicial] * n + [0.5]
    inferior = [-np.inf, -np.inf] + [0.0] * n + [SIGMA_MIN_MHZ]
    superior = [np.inf, np.inf] + [np.inf] * n + [SIGMA_MAX_MHZ]

    ajuste = least_squares(
        lambda p: (y - modelo_lineas(x, p, frecuencias)) / ruido,
        inicial, bounds=(inferior, superior), max_nfev=2000
    )
    chi_lineas = np.sum(ajuste.fun**2)

    matriz = np.column_stack([np.ones(len(x)), x - centro_ref])
    coef, *_ = np.linalg.lstsq(matriz, y, rcond=None)
    chi_nulo = np.sum(((y - matriz @ coef) / ruido)**2)

    return ajuste, chi_nulo - chi_lineas


def frecuencias_observadas(reposo, velocidad):
    """Convención radio: v = c * (f_reposo - f_obs) / f_reposo."""
    return reposo * (1.0 - velocidad / C_KMS)


def barrido(x, y, ruido, reposo, velocidades):
    """Ajusta el modelo global a cada velocidad y devuelve una tabla."""
    filas = []
    for v in velocidades:
        try:
            ajuste, mejora = ajustar_modelo(
                x, y, ruido, frecuencias_observadas(reposo, v))
        except (ValueError, FloatingPointError):
            continue
        if ajuste.success and np.isfinite(mejora):
            filas.append({
                "V_LSR_km_s": v,
                "mejora_chi2": mejora,
                "sigma_MHz": ajuste.x[-1],
                "amplitudes": ajuste.x[2:2 + len(reposo)],
            })
    return pd.DataFrame(filas)


# ---------- Ajuste local por grupos de líneas ----------

def agrupar(pos, ancho):
    """Junta líneas cuyas posiciones están a menos de 2*ancho (se ajustan juntas)."""
    orden = np.argsort(pos)
    grupos, actual = [], [orden[0]]
    for i in orden[1:]:
        if pos[i] - pos[actual[-1]] > 2 * ancho:
            grupos.append(actual)
            actual = [i]
        else:
            actual.append(i)
    grupos.append(actual)
    return grupos


def ajustar_grupo(freq, flux, ruido, f0s, v_ref, tol_v, rapido=False, phi=0.0, grado=None):
    """Fondo lineal LOCAL + gaussianas, con un corrimiento de velocidad dv
    común al grupo (limitado a +-tol_v) y una anchura común."""
    n = len(f0s)
    g_base = grado or GRADO_BASE
    pos0 = frecuencias_observadas(f0s, v_ref)
    margen = tol_v / C_KMS * pos0.max()
    m = ((freq >= pos0.min() - LOCAL_MHZ - margen)
         & (freq <= pos0.max() + LOCAL_MHZ + margen))
    x, y = freq[m], flux[m]
    if len(x) < 2 * n + 8:
        return None
    xc = x.mean()
    paso = float(np.median(np.diff(x)))
    t = (x - xc) / LOCAL_MHZ

    def modelo(p):
        centros = frecuencias_observadas(f0s, v_ref + p[2])
        salida = p[0] + p[1] * t
        if g_base >= 2:
            salida = salida + p[-1] * t**2
        for amp, c in zip(p[4:4 + n], centros):
            salida = salida + amp * np.exp(-0.5 * ((x - c) / p[3])**2)
        return salida

    kw = np.sqrt(1.0 - phi**2)

    def blanquear(v):
        """Convierte el ruido correlacionado (AR(1)) en ruido independiente."""
        w = np.empty(len(v))
        w[0] = v[0]
        w[1:] = (v[1:] - phi * v[:-1]) / kw
        return w

    base0 = np.median(y)
    amp0 = []
    for c in pos0:
        cerca = np.abs(x - c) < 1.0
        amp0.append(max(y[cerca].max() - base0, ruido) if cerca.any() else ruido)

    extra_b = g_base - 1        # términos de curvatura del fondo
    inferior = [-np.inf, -np.inf, -tol_v, SIGMA_LOCAL_MIN_MHZ] + [0.0] * n + [-np.inf] * extra_b
    superior = [np.inf, np.inf, tol_v, SIGMA_LOCAL_MAX_MHZ] + [np.inf] * n + [np.inf] * extra_b

    mejor = None
    for dv0 in np.linspace(-0.7 * tol_v, 0.7 * tol_v, 3 if rapido else 5):
        for s0 in ((0.5,) if rapido else (0.3, 0.8)):
            p0 = [base0, 0.0, dv0, s0] + list(amp0) + [0.0] * extra_b
            try:
                r = least_squares(lambda p: blanquear(y - modelo(p)) / ruido, p0,
                                  bounds=(inferior, superior), max_nfev=300)
            except ValueError:
                continue
            if mejor is None or r.cost < mejor.cost:
                mejor = r
    if mejor is None:
        return None

    p = mejor.x
    try:
        cov = np.linalg.pinv(mejor.jac.T @ mejor.jac)
        sdv = float(np.sqrt(abs(cov[2, 2])))
    except np.linalg.LinAlgError:
        cov = np.zeros((4 + n + extra_b, 4 + n + extra_b))
        sdv = tol_v
    sdv = float(np.clip(sdv, 0.02, tol_v))

    # Área integrada de cada línea (Jy/beam km/s) y su error, propagado desde la covarianza
    cen = frecuencias_observadas(f0s, v_ref + p[2])
    kf = np.sqrt(2 * np.pi) * C_KMS / cen
    area = p[4:4 + n] * p[3] * kf
    s_area = np.empty(n)
    for k in range(n):
        var = ((p[3] * kf[k])**2 * cov[4 + k, 4 + k] + (p[4 + k] * kf[k])**2 * cov[3, 3]
               + 2 * p[3] * p[4 + k] * kf[k]**2 * cov[3, 4 + k])
        s_area[k] = np.sqrt(max(var, 1e-12))

    # Prueba F: ¿las líneas mejoran el ajuste frente a solo el fondo? (p-valor aproximado)
    chi_mod = 2.0 * mejor.cost
    matriz = np.column_stack([t**j for j in range(g_base + 1)])
    mat_b = np.column_stack([blanquear(matriz[:, j]) for j in range(matriz.shape[1])])
    y_b = blanquear(y)
    coef, *_ = np.linalg.lstsq(mat_b, y_b, rcond=None)
    chi_nulo = float(np.sum(((y_b - mat_b @ coef) / ruido)**2))
    extra, gl = n + 2, len(x) - (4 + n + extra_b)
    if gl > 0 and 0 < chi_mod < chi_nulo:
        p_f = float(dist_f.sf(((chi_nulo - chi_mod) / extra) / (chi_mod / gl), extra, gl))
    else:
        p_f = 1.0

    return {
        "x": x, "y": y, "mod": modelo(p), "dv": float(p[2]), "s": float(p[3]),
        "A": p[4:4 + n].copy(), "sdv": sdv, "paso": paso,
        "cen": frecuencias_observadas(f0s, v_ref + p[2]),
        "v": v_ref + float(p[2]), "p_f": p_f, "W": area, "sW": s_area,
    }


def medir_lineas(freq, flux, ruido, reposo, v_ref, tol_v, rapido=False, phi=0.0, grado=None):
    """Ajusta cada grupo de líneas cercanas por separado."""
    pos = frecuencias_observadas(reposo, v_ref)
    resultados = []
    for idx in agrupar(pos, LOCAL_MHZ):
        idx = np.array(idx)
        r = ajustar_grupo(freq, flux, ruido, reposo[idx], v_ref, tol_v, rapido, phi, grado)
        if r is not None:
            r["idx"] = idx
            resultados.append(r)
    return resultados


def mediana_ponderada(valores, pesos):
    valores, pesos = np.asarray(valores), np.asarray(pesos)
    orden = np.argsort(valores)
    acumulado = np.cumsum(pesos[orden])
    return float(valores[orden][np.searchsorted(acumulado, 0.5 * acumulado[-1])])


# ---------- Catálogo local, intensidad integrada, LTE y Einstein ----------

def buscar_catalogo(molecula, f0):
    clave = f"{molecula}|{f0:.4f}"
    return loc.loc[clave] if clave in loc.index else None


def factor_lte(r):
    """log de [10^logI / D300]: intensidad relativa de la línea (sin Q ni T)."""
    el_k = float(r["E_inferior_cm_1"]) * K_CM
    eu_k = float(r["E_superior_K"])
    d300 = np.exp(-el_k / 300.0) - np.exp(-eu_k / 300.0)
    return float(r["log10_I_300K_nm2_MHz"]) * np.log(10.0) - np.log(d300), eu_k


def ajuste_lte(w_obs, s_obs, lw, eu):
    """Busca T y escala N que mejor reproducen las áreas observadas.
    Intensidad integrada esperada (ópticamente delgado):  W ∝ 10^logI / D300 * exp(-Eu/T).
    Q(T) es común a toda la molécula y se cancela al comparar sus líneas.
    El ajuste es robusto (Huber): las líneas que no encajan no arrastran al resto,
    así el ajuste lo define la mayoría y las inconsistentes quedan a la vista."""
    s_fit = np.sqrt(s_obs**2 + (FRAC_LTE * w_obs)**2)
    mejor = None
    for T in np.geomspace(T_LTE_MIN, T_LTE_MAX, 60):
        w = np.exp(lw - eu / T)
        w = w / w.max()
        N = np.sum(w_obs * w / s_fit**2) / np.sum(w**2 / s_fit**2)
        for _ in range(10):
            r = (w_obs - N * w) / s_fit
            peso = 1.0 / np.maximum(1.0, np.abs(r) / 1.5)
            N = np.sum(peso * w_obs * w / s_fit**2) / np.sum(peso * w**2 / s_fit**2)
        r = (w_obs - N * w) / s_fit
        costo = np.sum(np.where(np.abs(r) <= 1.5, 0.5 * r**2, 1.5 * (np.abs(r) - 0.75)))
        if mejor is None or costo < mejor[0]:
            mejor = (costo, T, N * w)
    _, T, w_pred = mejor
    return T, w_pred


def einstein_A(r, q300):
    """A_ul (s^-1) a partir de log I, Q(300 K), energías, g_u y frecuencia.
    S*mu^2 = 2.40251e4 * 10^logI * Q / (nu * [exp(-El/kT) - exp(-Eu/kT)])   (D^2)
    A_ul   = 1.16395e-20 * nu^3 * S*mu^2 / g_u                              (nu en MHz)"""
    nu = float(r["frecuencia_reposo_MHz"])
    el_k = float(r["E_inferior_cm_1"]) * K_CM
    eu_k = float(r["E_superior_K"])
    d300 = np.exp(-el_k / 300.0) - np.exp(-eu_k / 300.0)
    smu2 = 2.40251e4 * 10**float(r["log10_I_300K_nm2_MHz"]) * q300 / (nu * d300)
    return 1.16395e-20 * nu**3 * smu2 / float(r["g_superior"])


def ruido_canales_libres(y):
    """Desviación estándar de los canales sin señal (recorte iterativo a 3 sigma)."""
    y0 = np.asarray(y, dtype=float) - np.median(y)
    sigma = np.std(y0)
    for _ in range(6):
        sigma = np.std(y0[np.abs(y0) < 3 * sigma])
    return float(sigma)


def cumple_amplitud(r, k, ruido):
    """SNR de pico (por canal), SNR integrado y prueba F de la línea k de un grupo ajustado."""
    return r["A"][k] / ruido, r["W"][k] / r["sW"][k], r["p_f"]


def control_falsas_alarmas(freq, flux, ruido, reposo, v_med, phi=0.0):
    """Desplaza las posiciones esperadas a zonas donde no está la molécula y cuenta cuántas
    'líneas' cumplirían los criterios de amplitud (estrictos y de tentativa).
    Devuelve (aciertos estrictos, total, estrictos lejos del catálogo,
              aciertos tentativos, tentativos lejos del catálogo).
    'Lejos del catálogo' = no hay ninguna línea catalogada a menos de RADIO_CATALOGO_MHZ:
    esa es la tasa de falsas alarmas por ruido o líneas desconocidas. La tasa total es una
    cota alta porque incluye líneas reales de otras especies."""
    rng = np.random.default_rng(0)
    cat_obs = np.sort(frecuencias_observadas(cat["frecuencia_reposo_MHz"].to_numpy(float), v_med))

    def cerca_de_catalogo(f):
        j = np.searchsorted(cat_obs, f)
        d = np.inf
        if j < len(cat_obs):
            d = min(d, abs(cat_obs[j] - f))
        if j > 0:
            d = min(d, abs(cat_obs[j - 1] - f))
        return d <= RADIO_CATALOGO_MHZ

    total = ok = ok_libre = ok_t = ok_t_libre = 0
    for _ in range(N_CONTROL):
        rep_ = reposo + rng.uniform(40, 150) * rng.choice([-1, 1])
        obs = frecuencias_observadas(rep_, v_med)
        rep_ = rep_[(obs > freq.min() + LOCAL_MHZ + 2) & (obs < freq.max() - LOCAL_MHZ - 2)]
        if len(rep_) == 0:
            continue
        for r in medir_lineas(freq, flux, ruido, rep_, v_med, TOL_FINAL_KMS, rapido=True, phi=phi):
            for k in range(len(r["idx"])):
                total += 1
                snr, snr_i, p_f = cumple_amplitud(r, k, ruido)
                libre = not cerca_de_catalogo(r["cen"][k])
                if snr >= SIGMA_DETECCION and snr_i >= SNR_INTEGRADO_MIN and p_f < P_VALOR_F_MAX:
                    ok += 1
                    ok_libre += int(libre)
                if snr >= SIGMA_TENTATIVA and snr_i >= SNR_INTEGRADO_MIN and p_f < P_VALOR_F_TENT:
                    ok_t += 1
                    ok_t_libre += int(libre)
    return ok, total, ok_libre, ok_t, ok_t_libre


def pruebas_internas(freq, ruido, phi):
    """Ruido plano: no debe haber detecciones. Líneas inyectadas: deben recuperarse."""
    rng = np.random.default_rng(1)
    reposo = np.array([349120.0, 349160.0, 349240.0])
    falsas = lineas = 0
    for _ in range(5):
        y = ruido_ar1(len(freq), ruido, phi, rng)
        for r in medir_lineas(freq, y, ruido, reposo, 3.0, TOL_FINAL_KMS, rapido=True, phi=phi):
            for k in range(len(r["idx"])):
                lineas += 1
                snr, snr_i, p_f = cumple_amplitud(r, k, ruido)
                falsas += int(snr >= SIGMA_DETECCION and snr_i >= SNR_INTEGRADO_MIN
                              and p_f < P_VALOR_F_MAX)
    print(f"Prueba 1 (ruido plano): {falsas} detecciones falsas en {lineas} líneas -> "
          + ("OK" if falsas == 0 else "REVISAR"))

    s_i, a_i, v_i = 0.55, 8 * ruido, 3.0
    y = ruido_ar1(len(freq), ruido, phi, rng)
    for f0 in reposo:
        y = y + a_i * np.exp(-0.5 * ((freq - f0 * (1 - v_i / C_KMS)) / s_i)**2)
    grupos = medir_lineas(freq, y, ruido, reposo, 3.4, TOL_EXPLORA_KMS, phi=phi)
    errores_v = [abs(r["v"] - v_i) for r in grupos]
    errores_a = [abs(r["A"][k] - a_i) / a_i for r in grupos for k in range(len(r["idx"]))]
    bien = (len(errores_a) == 3 and max(errores_v) < 0.15 and max(errores_a) < 0.2)
    print(f"Prueba 2 (3 líneas inyectadas a {v_i} km/s, 8 sigma): "
          f"error máx. en velocidad {max(errores_v):.2f} km/s, en amplitud "
          f"{100 * max(errores_a):.0f}% -> " + ("OK" if bien else "REVISAR"))


def aplicar_velocidad_sistemica(df):
    """Reevalúa el estado de cada línea comparando su velocidad con la de la FUENTE (y no con
    la velocidad estimada de cada molécula, que se contamina con coincidencias).
    Todas las moléculas de una misma fuente comparten (casi) la misma velocidad. Una línea con
    SNR alto pero a otra velocidad es casi seguro una coincidencia con otra línea del bosque
    espectral. La velocidad sistémica es la mediana de las líneas detectadas de las moléculas con
    al menos MIN_LINEAS_ANCLA líneas (o V_SISTEMICA si se fijó a mano).
    Niveles: 'detectada' (criterios estrictos), 'tentativa' (más débil, pero con velocidad,
    fondo y ancho de línea consistentes), 'dudosa' y 'no detectada'."""
    antes = df["estado"].copy()
    det = df[df["estado"] == "detectada"]
    if V_SISTEMICA is not None:
        v_sis = float(V_SISTEMICA)
        anc = det
    else:
        n_det = det.groupby("molecula")["V_linea_km_s"].transform("count")
        anc = det[n_det >= MIN_LINEAS_ANCLA]
        if anc["V_linea_km_s"].dropna().empty:
            print("\nAVISO: ninguna molécula tiene líneas detectadas suficientes para estimar "
                  "la velocidad sistémica; no se aplica ese filtro.")
            df["dv_sistemica_km_s"] = np.nan
            df["ancho_relativo"] = np.nan
            return df, np.nan
        v_sis = float(anc["V_linea_km_s"].dropna().median())
    ancho_ref = float(anc["sigma_km_s"].median()) if len(anc) else np.nan
    df["dv_sistemica_km_s"] = (df["V_linea_km_s"] - v_sis).round(2)
    df["ancho_relativo"] = ((df["sigma_km_s"] / ancho_ref).round(2)
                            if np.isfinite(ancho_ref) else np.nan)

    motivo_viejo = df["motivo"].fillna("").astype(str)
    snr, snr_i, p_f = df["amplitud_sobre_ruido"], df["snr_integrado"], df["p_valor_F"]
    lte_mal = motivo_viejo.str.contains("intensidades LTE")
    borde = motivo_viejo.str.contains("límite del rango")
    inc_mal = (df["incertidumbre_MHz"] > INC_MAX_MHZ).fillna(False)
    dv = df["dv_sistemica_km_s"].abs()
    dv_ok_d, dv_ok_t = dv <= TOL_V_SISTEMICA_KMS, dv <= TOL_V_TENT_KMS
    ancho_ok = (((df["ancho_relativo"] - 1).abs() <= TOL_ANCHO)
                if np.isfinite(ancho_ref) else pd.Series(True, index=df.index))
    fondo_d = df["robusta_al_fondo"] == "sí"
    fondo_t = df["robusta_tentativa"] == "sí"
    lte_mal_t = lte_mal & (df["desviacion_LTE"].abs() > PULL_LTE_TENT)
    comun = (snr_i >= SNR_INTEGRADO_MIN) & ~borde & ~inc_mal
    estricta = (comun & ~lte_mal & (snr >= SIGMA_DETECCION) & (p_f < P_VALOR_F_MAX)
                & fondo_d & dv_ok_d)
    tentativa = (~estricta & comun & ~lte_mal_t & (snr >= SIGMA_TENTATIVA)
                 & (p_f < P_VALOR_F_TENT) & fondo_t & dv_ok_t & ancho_ok)
    nuevo = np.where(snr < SIGMA_EXPLORATORIO, "no detectada",
                     np.where(estricta, "detectada", np.where(tentativa, "tentativa", "dudosa")))
    df["estado"] = nuevo

    motivos = []
    for i in range(len(df)):
        if nuevo[i] in ("detectada", "no detectada"):
            motivos.append(motivo_viejo.iloc[i] if nuevo[i] == "no detectada" else "")
            continue
        m = []
        if snr.iloc[i] < SIGMA_DETECCION:
            m.append(f"SNR pico < {SIGMA_DETECCION:g}")
        if snr_i.iloc[i] < SNR_INTEGRADO_MIN:
            m.append(f"SNR integrado < {SNR_INTEGRADO_MIN:g}")
        if p_f.iloc[i] >= P_VALOR_F_MAX:
            m.append("prueba F")
        if not fondo_d.iloc[i]:
            m.append("depende del fondo")
        if lte_mal.iloc[i]:
            m.append("intensidades LTE")
        if borde.iloc[i]:
            m.append("velocidad en el límite del rango")
        if inc_mal.iloc[i]:
            m.append(f"posición del catálogo incierta (+-{df['incertidumbre_MHz'].iloc[i]:g} MHz)")
        tol = TOL_V_TENT_KMS if nuevo[i] == "tentativa" else TOL_V_SISTEMICA_KMS
        if not (dv.iloc[i] <= tol):
            m.append("velocidad distinta de la sistémica")
        if nuevo[i] == "dudosa" and not ancho_ok.iloc[i] and dv_ok_t.iloc[i]:
            m.append("ancho de línea distinto al de las líneas ancla")
        motivos.append("; ".join(m))
    df["motivo"] = motivos

    print(f"\nVelocidad sistémica usada: {v_sis:.2f} km/s (tolerancia +-{TOL_V_SISTEMICA_KMS:g} km/s "
          f"detectadas, +-{TOL_V_TENT_KMS:g} tentativas) | ancho de referencia: {ancho_ref:.2f} km/s")
    cambio = (antes != df["estado"]) & ((antes != "no detectada") | (df["estado"] != "no detectada"))
    if cambio.any():
        t = df.loc[cambio, ["ventana", "molecula", "reposo_MHz", "amplitud_sobre_ruido",
                            "dv_sistemica_km_s", "ancho_relativo"]].copy()
        t.insert(3, "antes", antes[cambio])
        t.insert(4, "ahora", df.loc[cambio, "estado"])
        print("Líneas cuyo estado cambió al comparar con la velocidad de la fuente:")
        print(t.sort_values(["molecula", "reposo_MHz"]).to_string(index=False))
    return df, v_sis


def evaluar_mezclas_y_moleculas(detalle, resumen):
    """Marca las líneas que otra molécula también reclama (mezclas) y decide el estado final
    de cada molécula usando solo líneas sin mezcla.
    Una línea detectada está mezclada si otra molécula analizada tiene una línea con SNR >= 3
    a menos de TOL_MEZCLA_CANALES canales. Si ambas tienen modelo LTE, se reparte el pico según
    las áreas esperadas y la línea se considera limpia si a esta molécula le toca >= FRAC_DOMINANTE."""
    df = pd.DataFrame(detalle).reset_index(drop=True)
    df, v_sis = aplicar_velocidad_sistemica(df)
    res = pd.DataFrame(resumen)
    canal = {v: float(np.median(np.diff(datos[v][0]))) for v in datos}
    reclaman = df[df["amplitud_sobre_ruido"] >= SIGMA_EXPLORATORIO]

    mezcla, fraccion, limpia, final = [], [], [], []
    for r in df.itertuples():
        if r.estado not in ("detectada", "tentativa"):
            mezcla.append(""); fraccion.append(np.nan); limpia.append(np.nan)
            final.append(r.estado)
            continue
        otras = reclaman[(reclaman["ventana"] == r.ventana)
                         & (reclaman["molecula"] != r.molecula)
                         & ((reclaman["observada_predicha_MHz"] - r.observada_predicha_MHz).abs()
                            <= TOL_MEZCLA_CANALES * canal[r.ventana])]
        if otras.empty:
            mezcla.append(""); fraccion.append(np.nan); limpia.append(True)
            final.append(r.estado)
            continue
        a_otras = otras["area_esperada_LTE"]
        if np.isfinite(r.area_esperada_LTE) and a_otras.notna().all():
            total = r.area_esperada_LTE + a_otras.sum()
            frac = r.area_esperada_LTE / total if total > 0 else np.nan
        else:
            frac = np.nan
        ok = bool(np.isfinite(frac) and frac >= FRAC_DOMINANTE)
        mezcla.append(", ".join(sorted(set(otras["molecula"]))))
        fraccion.append(frac); limpia.append(ok)
        final.append(r.estado if ok else r.estado + " con mezcla")
    df["mezcla_con"], df["fraccion_propia_LTE"] = mezcla, np.round(fraccion, 2)
    df["limpia"], df["estado_final"] = limpia, final

    filas_mol = []
    for molecula, g in df.groupby("molecula"):
        det = g[g["estado"] == "detectada"]
        lim = det[det["limpia"] == True]
        indep = {}
        for v, h in lim.groupby("ventana"):
            pos = np.sort(h["observada_predicha_MHz"].values)
            indep[v] = int(sum(1 for k, p in enumerate(pos)
                               if k == 0 or p - pos[k - 1] > canal[v]))
        n_lim = sum(indep.values())
        n_tent_all = int((g["estado"] == "tentativa").sum())
        tent_ok = g[(g["estado"] == "tentativa") & (g["limpia"] == True)]
        n_tent = 0
        for v, h in tent_ok.groupby("ventana"):
            pos = np.sort(h["observada_predicha_MHz"].values)
            n_tent += sum(1 for k, p in enumerate(pos) if k == 0 or p - pos[k - 1] > canal[v])
        r_m = res[res["molecula"] == molecula].set_index("ventana")
        ven_det = [v for v, n in indep.items() if n > 0]
        vw = {v: round(float(det[det["ventana"] == v]["V_linea_km_s"].median()), 2)
              for v in ven_det}
        if len(ven_det) == 2:
            dif = abs(vw[ven_det[0]] - vw[ven_det[1]])
            concuerdan = ("sí" if dif <= CANALES_COHERENCIA * max(r_m.loc[ven_det, "canal_km_s"])
                          else "no")
        else:
            concuerdan = "n/a"
        if n_lim >= MIN_LIMPIAS:
            if len(ven_det) == 2:
                estado = "confirmada" if concuerdan == "sí" else "inconsistente"
            else:
                estado = "probable"
        elif len(det) >= 1:
            estado = "candidata"
        elif n_tent >= MIN_TENTATIVAS:
            estado = "tentativa"
        elif n_tent_all >= MIN_TENTATIVAS:
            estado = "tentativa (con mezcla)"
        else:
            estado = "no detectada"
        if concuerdan == "no":
            print(f"AVISO: W1 y W2 dan velocidades distintas para {molecula} "
                  f"({vw[ven_det[0]]} y {vw[ven_det[1]]} km/s).")
        v_ref = list(vw.values())
        filas_mol.append({
            "molecula": molecula,
            "estado": estado,
            "V_LSR_km_s": round(float(np.median(v_ref)), 2) if v_ref else np.nan,
            "lineas_totales": int(len(g)),
            "lineas_detectadas": int(len(det)),
            "lineas_sin_mezcla": int(len(lim)),
            "lineas_con_mezcla": int(len(det) - len(lim)),
            "lineas_tentativas": n_tent_all,
            "tentativas_sin_mezcla": int(n_tent),
            "transiciones_sin_mezcla": int(n_lim),
            "ventanas_con_deteccion": len(ven_det),
            "ventanas_concuerdan": concuerdan,
            "falsas_esperadas": round(float(np.nansum(r_m["falsas_esperadas"])), 2),
            "falsas_esperadas_tentativas": round(float(np.nansum(r_m["falsas_esperadas_tentativas"])), 2),
        })
    n_an = df["molecula"].nunique()
    n_cat = cat[cat["ventana"].isin(list(ESPECTROS))]["especie"].nunique()
    if n_an < n_cat:
        print(f"\nAVISO: las mezclas solo se revisaron contra las {n_an} moléculas analizadas "
              f"(el catálogo tiene {n_cat}). Escribe TODAS para revisarlas contra todas.")
    return df, pd.DataFrame(filas_mol)


def con_huecos(x, y):
    """Inserta NaN donde faltan canales para que la gráfica no una tramos lejanos."""
    salto = np.where(np.diff(x) > 3 * np.median(np.diff(x)))[0] + 1
    return np.insert(x, salto, np.nan), np.insert(y, salto, np.nan)


def grafica_snr(lineas, estados_mol, archivo="evidencias_procesos/snr_lineas.png"):
    """Amplitud sobre ruido de cada línea detectada, por frecuencia de reposo y molécula.
    Círculo lleno: línea sin mezcla. Círculo hueco: detectada pero mezclada con otra molécula.
    Línea punteada negra: umbral de detección (5 sigma). Gris: 3 sigma (exploratorio)."""
    df = pd.DataFrame(lineas)
    validos = ["detectada", "detectada con mezcla", "tentativa", "tentativa con mezcla"]
    if not GRAFICA_SOLO_DETECTADAS:
        validos.append("dudosa")
    df = df[df["estado_final"].isin(validos)]
    if df.empty:
        print("No hay líneas para graficar el SNR.")
        return

    ventanas = sorted(df["ventana"].unique())
    colores = {m: plt.get_cmap("tab20")(((i % 10) * 2 + i // 10) % 20)
               for i, m in enumerate(sorted(df["molecula"].unique()))}

    fig, axes = plt.subplots(len(ventanas), 1, figsize=(11, 4 * len(ventanas)),
                             squeeze=False)
    for ax, ventana in zip(axes[:, 0], ventanas):
        sub = df[df["ventana"] == ventana]
        for molecula, g in sub.groupby("molecula"):
            etiqueta = f"{molecula} ({estados_mol.get(molecula, '')})"
            usada = False
            for estado, h in g.groupby("estado_final"):
                hueca = "mezcla" in estado
                alpha = 1.0 if estado == "detectada" else 0.75 if estado.startswith("detectada") else 0.6
                ax.vlines(h["reposo_MHz"], 0, h["amplitud_sobre_ruido"],
                          color=colores[molecula], lw=2, alpha=alpha)
                ax.scatter(h["reposo_MHz"], h["amplitud_sobre_ruido"],
                           facecolors="white" if hueca else colores[molecula],
                           edgecolors=colores[molecula], linewidths=1.5, alpha=alpha,
                           marker="s" if estado.startswith("tentativa") else "o",
                           zorder=3, label=None if usada else etiqueta)
                usada = True
        ax.axhline(SIGMA_EXPLORATORIO, color="gray", linestyle="--", lw=1,
                   label=f"{SIGMA_EXPLORATORIO:g} sigma (exploratorio)")
        ax.axhline(SIGMA_TENTATIVA, color="tab:orange", linestyle="-.", lw=1.2,
                   label=f"{SIGMA_TENTATIVA:g} sigma (tentativa)")
        ax.axhline(SIGMA_DETECCION, color="black", linestyle=":", lw=1.8,
                   label=f"{SIGMA_DETECCION:g} sigma (detección)")
        ax.set_title(f"Líneas detectadas: amplitud sobre ruido | Ventana: {ventana}"
                     "  (hueco = mezclada; cuadrado = tentativa)")
        ax.set_xlabel("Frecuencia de reposo (MHz)")
        ax.set_ylabel("Amplitud / ruido (SNR)")
        ax.set_ylim(bottom=0)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(archivo, dpi=300, bbox_inches="tight")
    print(f"Gráfica de SNR guardada en: {archivo}")
    plt.show()


# ============== CARGA INICIAL Y AUDITORÍA TÉCNICA ==============

os.makedirs("evidencias_procesos", exist_ok=True)
datos = {}
PHI = {}
auditoria_ruido = []

for ventana, archivo in ESPECTROS.items():
    if not os.path.exists(archivo):
        posibles = [f for f in os.listdir('.') if ventana in f and f.endswith('.csv')]
        if posibles:
            archivo = posibles[0]

    df = pd.read_csv(archivo, usecols=["frecuencia_LSRK_GHz", "B_offset_Jy_beam"])
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    df = df.sort_values("frecuencia_LSRK_GHz")

    freq = df["frecuencia_LSRK_GHz"].to_numpy() * 1000.0    # GHz -> MHz
    flux = df["B_offset_Jy_beam"].to_numpy()
    ruido, phi_v = estimar_ruido_correlacionado(flux)
    PHI[ventana] = phi_v
    datos[ventana] = (freq, flux, ruido)

    auditoria_ruido.append({
        "ventana": ventana,
        "archivo_fuente": archivo,
        "total_canales": len(freq),
        "frecuencia_min_MHz": float(freq.min()),
        "frecuencia_max_MHz": float(freq.max()),
        "ruido_MAD_sin_corregir_Jy_beam": round(estimar_ruido(flux), 5),
        "autocorrelacion_canales_vecinos": round(phi_v, 2),
        "ruido_corregido_usado_Jy_beam": round(ruido, 5),
        "ruido_std_canales_libres_Jy_beam": round(ruido_canales_libres(flux), 5),
    })
    print(f"{ventana}: {len(freq)} canales | ruido por diferencias = {estimar_ruido(flux):.4f} | "
          f"correlación entre vecinos = {phi_v:.2f} | ruido corregido (usado) = {ruido:.4f} | "
          f"desviación de canales libres = {ruido_canales_libres(flux):.4f} Jy/beam")

pd.DataFrame(auditoria_ruido).to_csv("evidencias_procesos/auditoria_ruido_espectros.csv", index=False)

cat = pd.read_csv(CATALOGO)

loc = pd.read_csv(CATALOGO_LOCAL, dtype={"QN_superior": str, "QN_inferior": str})
loc["clave"] = (loc["especie"] + "|"
                + loc["frecuencia_reposo_MHz"].round(4).map("{:.4f}".format))
loc = loc.drop_duplicates("clave").set_index("clave")

Q300 = {}
if os.path.exists(PARTICION):
    q = pd.read_csv(PARTICION).dropna(subset=["Q_300K"])
    Q300 = dict(zip(q["especie"], q["Q_300K"].astype(float)))
    if Q300:
        print(f"Función de partición cargada para {len(Q300)} especies "
              "(se calculará el coeficiente de Einstein en ellas).")
    else:
        print(f"'{PARTICION}' no tiene valores de Q_300K: no se calcula el coeficiente de Einstein.")
else:
    print(f"Sin '{PARTICION}': no se calcula el coeficiente de Einstein "
          "(el resto del análisis funciona igual).")

resumen, detalle, bitacora = [], [], []
estados_mol = {}
cola, automatico = [], False

if EJECUTAR_PRUEBAS:
    print()
    pruebas_internas(datos['W1'][0], datos['W1'][2], PHI['W1'])


# ============== INTERACCIÓN CON EL USUARIO ==============

while True:

    if cola:
        molecula, texto, automatico = cola.pop(0), "", True
        print(f"\n>>> Analizando {molecula} ...")
    else:
        automatico = False
        molecula = input("\nNombre de la molécula (Enter para terminar; TODAS = todas las del "
                         "catálogo): ").strip()
        if not molecula:
            break
        if molecula.upper() == "TODAS":
            cola = sorted(cat[cat["ventana"].isin(list(ESPECTROS))]["especie"].unique())
            continue
        texto = input("Frecuencias de reposo en MHz, separadas por comas "
                      "(Enter = usar el catálogo): ")
    try:
        manuales = np.array([float(x) for x in texto.split(",") if x.strip()])
    except ValueError:
        print("ERROR: escribe solo números separados por comas.")
        continue
    if np.any(manuales <= 0):
        print("ERROR: las frecuencias deben ser positivas.")
        continue

    filas, lineas_mol, zoom = [], [], []

    # Panorámica de los espectros completos con el nivel de ruido
    fig_pan, axes_pan = plt.subplots(2, 1, figsize=(12, 6))
    for i, (win_name, (f_arr, fl_arr, r_val)) in enumerate(datos.items()):
        ax_p = axes_pan[i]
        ax_p.plot(f_arr, fl_arr, color='gray', lw=0.8, label=f'Espectro {win_name}')
        ax_p.axhline(r_val, color='red', linestyle='--', label=f'±1 sigma (ruido corregido) = {r_val:.4f}')
        ax_p.axhline(-r_val, color='red', linestyle='--')
        ax_p.set_title(f"Diagnóstico de Procesos: Panorámica de Espectros y Ruido [{win_name}]")
        ax_p.set_xlabel("Frecuencia (MHz)")
        ax_p.set_ylabel("Flujo (Jy/beam)")
        ax_p.legend(loc="upper right")
        ax_p.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"evidencias_procesos/panoramica_{molecula}.png", dpi=300)
    plt.close()

    fig, axes = plt.subplots(len(datos), 1, figsize=(10, 4 * len(datos)), squeeze=False)
    ax_idx = 0

    for ventana, (freq, flux, ruido) in datos.items():
        phi = PHI[ventana]

        if len(manuales) > 0:
            reposo = np.sort(np.unique(manuales))
        else:
            lineas = cat[(cat["especie"] == molecula) & (cat["ventana"] == ventana)]
            reposo = np.sort(lineas["frecuencia_reposo_MHz"].round(4).unique())

        reposo = reposo[
            (frecuencias_observadas(reposo, V_MAX) >= freq.min())
            & (frecuencias_observadas(reposo, V_MIN) <= freq.max())
        ]

        ax = axes[ax_idx, 0]
        ax_idx += 1

        if len(reposo) == 0:
            print(f"{ventana}: no hay líneas de {molecula} en esta ventana.")
            ax.set_visible(False)
            continue

        # ---- Paso 1: velocidad inicial con el ajuste global (como antes) ----
        mascara = np.zeros(len(freq), dtype=bool)
        for f0 in reposo:
            mascara |= (
                (freq >= frecuencias_observadas(f0, V_MAX) - VENTANA_MHZ)
                & (freq <= frecuencias_observadas(f0, V_MIN) + VENTANA_MHZ)
            )
        x, y = freq[mascara], flux[mascara]
        if len(x) < 2 * len(reposo) + 5:
            print(f"{ventana}: muy pocos canales para ajustar {len(reposo)} líneas.")
            ax.set_visible(False)
            continue

        grueso = barrido(x, y, ruido, reposo, np.arange(V_MIN, V_MAX + PASO_V / 2, PASO_V))
        if grueso.empty:
            print(f"{ventana}: no se pudo ajustar el perfil.")
            ax.set_visible(False)
            continue
        v0 = grueso.loc[grueso["mejora_chi2"].idxmax(), "V_LSR_km_s"]
        fino = barrido(x, y, ruido, reposo, np.arange(
            max(V_MIN, v0 - PASO_V), min(V_MAX, v0 + PASO_V) + PASO_REFINAMIENTO / 2,
            PASO_REFINAMIENTO))
        mejor = (fino if not fino.empty else grueso)
        v_global = float(mejor.loc[mejor["mejora_chi2"].idxmax(), "V_LSR_km_s"])

        # ---- Paso 2: cada grupo de líneas se ajusta con su propio fondo y velocidad ----
        pasada1 = medir_lineas(freq, flux, ruido, reposo, v_global, TOL_EXPLORA_KMS, phi=phi)
        def posicion_fiable(r):
            """La línea más fuerte del grupo tiene posición bien conocida en el catálogo."""
            inf = buscar_catalogo(molecula, reposo[r["idx"][int(np.argmax(r["A"]))]])
            return inf is None or not (float(inf["incertidumbre_MHz"]) > INC_MAX_MHZ)
        fiables = [r for r in pasada1 if posicion_fiable(r)] or pasada1
        fuertes = [r for r in fiables if r["A"].max() / ruido >= SIGMA_DETECCION]
        if not fuertes:
            fuertes = [r for r in fiables if r["A"].max() / ruido >= SIGMA_EXPLORATORIO]
        if fuertes:
            v_med = mediana_ponderada([r["v"] for r in fuertes],
                                      [1.0 / r["sdv"]**2 for r in fuertes])
            dispersion = (float(np.std([r["v"] for r in fuertes]))
                          if len(fuertes) > 1 else np.nan)
        else:
            v_med, dispersion = v_global, np.nan
        v1_linea = {}
        for r in pasada1:
            for i in r["idx"]:
                v1_linea[i] = (r["v"], r["sdv"])

        # ---- Paso 3: medida final cerca de la velocidad robusta ----
        pasada2 = medir_lineas(freq, flux, ruido, reposo, v_med, TOL_FINAL_KMS, phi=phi)
        # Robustez: la misma medida con el otro tipo de fondo (recta <-> parábola)
        pasada_alt = medir_lineas(freq, flux, ruido, reposo, v_med, TOL_FINAL_KMS, phi=phi,
                                  grado=1 if GRADO_BASE >= 2 else 2)
        robusta, robusta_t = {}, {}
        for r in pasada_alt:
            for k, i in enumerate(r["idx"]):
                snr_a, snr_i_a, p_a = cumple_amplitud(r, k, ruido)
                robusta[i] = bool(snr_a >= SIGMA_DETECCION and snr_i_a >= SNR_INTEGRADO_MIN
                                  and p_a < P_VALOR_F_MAX)
                robusta_t[i] = bool(snr_a >= SIGMA_TENTATIVA and snr_i_a >= SNR_INTEGRADO_MIN
                                    and p_a < P_VALOR_F_TENT)
        if not pasada2:
            print(f"{ventana}: no se pudo medir ninguna línea.")
            ax.set_visible(False)
            continue

        medidas = []   # una entrada por línea
        for r in pasada2:
            for k, i in enumerate(r["idx"]):
                a, s, f_obs = r["A"][k], r["s"], r["cen"][k]
                factor = np.sqrt(2 * np.pi) * C_KMS / f_obs
                s_a = ruido / np.sqrt(s * np.sqrt(np.pi) / r["paso"])
                v1, sdv1 = v1_linea.get(i, (np.nan, np.nan))
                medidas.append({
                    "i": i, "f0": reposo[i], "obs": f_obs, "A": a, "snr": a / ruido,
                    "W": r["W"][k], "sW": r["sW"][k], "snr_int": r["W"][k] / r["sW"][k],
                    "p_f": r["p_f"], "canal_kms": r["paso"] * C_KMS / f_obs,
                    "robusta": robusta.get(i, False), "robusta_t": robusta_t.get(i, False),
                    "s": s,
                    "v1": v1, "sdv1": sdv1, "info": buscar_catalogo(molecula, reposo[i]),
                })
        medidas.sort(key=lambda m: m["i"])

        # ---- Paso 4: chequeo de intensidades relativas (LTE) ----
        T_ex = np.nan
        for m in medidas:
            m["pull"], m["W_lte"] = np.nan, np.nan
        con_cat = [m for m in medidas if m["info"] is not None]
        if (len(con_cat) >= 2 and any(m["snr"] >= SIGMA_EXPLORATORIO for m in con_cat)):
            lw, eu = np.array([factor_lte(m["info"]) for m in con_cat]).T
            w_obs = np.array([m["W"] for m in con_cat])
            s_obs = np.array([m["sW"] for m in con_cat])
            T_ex, w_pred = ajuste_lte(w_obs, s_obs, lw, eu)
            for m, wp in zip(con_cat, w_pred):
                s_ef = np.sqrt(m["sW"]**2 + (FRAC_LTE * min(m["W"], wp))**2)
                m["pull"], m["W_lte"] = (m["W"] - wp) / s_ef, wp

        # El chequeo LTE solo es útil si la mayoría de las líneas fuertes encaja;
        # si casi ninguna encaja, el modelo no describe estos datos y no se usa.
        fuertes_lte = [m for m in medidas
                       if m["snr"] >= SIGMA_EXPLORATORIO and np.isfinite(m["pull"])]
        lte_concluyente = bool(fuertes_lte) and (
            np.mean([abs(m["pull"]) <= PULL_LTE for m in fuertes_lte]) >= FRAC_LTE_MINIMA)
        if not lte_concluyente:
            T_ex = np.nan

        # ---- Estado de cada línea (criterios de detección) ----
        borde = v_med <= V_MIN + 0.2 or v_med >= V_MAX - 0.2
        for m in medidas:
            coherente = (np.isfinite(m["v1"]) and
                         abs(m["v1"] - v_med) <= max(CANALES_COHERENCIA * m["canal_kms"],
                                                     3 * m["sdv1"]))
            lte_ok = ((not lte_concluyente) or (not np.isfinite(m["pull"]))
                      or abs(m["pull"]) <= PULL_LTE)
            motivos = []
            if m["snr"] < SIGMA_DETECCION:
                motivos.append(f"SNR pico < {SIGMA_DETECCION:g}")
            if m["snr_int"] < SNR_INTEGRADO_MIN:
                motivos.append(f"SNR integrado < {SNR_INTEGRADO_MIN:g}")
            if m["p_f"] >= P_VALOR_F_MAX:
                motivos.append("prueba F")
            if not coherente:
                motivos.append("velocidad")
            if not lte_ok:
                motivos.append("intensidades LTE")
            if not m["robusta"]:
                motivos.append("depende del fondo")
            if borde:
                motivos.append("velocidad en el límite del rango")
            inc = (float(m["info"]["incertidumbre_MHz"]) if m["info"] is not None else np.nan)
            m["inc"] = inc
            pos_incierta = bool(np.isfinite(inc) and inc > INC_MAX_MHZ)
            if pos_incierta:
                motivos.append(f"posición del catálogo incierta (+-{inc:g} MHz)")
            # Nivel tentativa: mismos criterios, con los umbrales de amplitud más bajos
            motivos_t = []
            if m["snr"] < SIGMA_TENTATIVA:
                motivos_t.append("SNR pico")
            if m["snr_int"] < SNR_INTEGRADO_MIN:
                motivos_t.append("SNR integrado")
            if m["p_f"] >= P_VALOR_F_TENT:
                motivos_t.append("prueba F")
            if not coherente:
                motivos_t.append("velocidad")
            if not lte_ok:
                motivos_t.append("intensidades LTE")
            if not m["robusta_t"]:
                motivos_t.append("depende del fondo")
            if borde or pos_incierta:
                motivos_t.append("borde o catálogo")
            m["coherente"], m["lte_ok"], m["motivo"] = coherente, lte_ok, "; ".join(motivos)
            if m["snr"] < SIGMA_EXPLORATORIO:
                m["estado"] = "no detectada"
            elif not motivos:
                m["estado"] = "detectada"
            elif not motivos_t:
                m["estado"] = "tentativa"
            else:
                m["estado"] = "dudosa"

        n3 = sum(m["snr"] >= SIGMA_EXPLORATORIO for m in medidas)
        nd = sum(m["estado"] == "detectada" for m in medidas)
        nt = sum(m["estado"] == "tentativa" for m in medidas)
        # líneas que el catálogo predice fuertes (según el ajuste LTE) y no aparecen
        ausentes = (sum(1 for m in medidas if m["snr"] < SIGMA_EXPLORATORIO
                        and np.isfinite(m["pull"]) and m["pull"] < -PULL_LTE)
                    if lte_concluyente else 0)
        paso_canal = pasada2[0]["paso"]
        det_f = sorted(m["obs"] for m in medidas if m["estado"] == "detectada")
        indep = sum(1 for k, fv in enumerate(det_f) if k == 0 or fv - det_f[k - 1] > paso_canal)
        ok_c, tot_c, ok_l, ok_t, ok_tl = control_falsas_alarmas(freq, flux, ruido, reposo, v_med, phi)
        tasa_fa = ok_c / tot_c if tot_c else np.nan
        tasa_fa_libre = ok_l / tot_c if tot_c else np.nan
        tasa_t_libre = ok_tl / tot_c if tot_c else np.nan
        filas.append({
            "ventana": ventana,
            "molecula": molecula,
            "lineas": len(medidas),
            "V_LSR_km_s": round(v_med, 2),
            "V_global_km_s": round(v_global, 2),
            "dispersion_V_km_s": round(dispersion, 2) if np.isfinite(dispersion) else np.nan,
            "lineas_3sigma": int(n3),
            "transiciones_independientes": int(indep),
            "lineas_detectadas": int(nd),
            "lineas_tentativas": int(nt),
            "lineas_dudosas": int(n3 - nd - nt),
            "esperadas_ausentes": int(ausentes),
            "T_ex_K": round(T_ex) if np.isfinite(T_ex) else np.nan,
            "sigma_MHz": round(float(np.median([r["s"] for r in pasada2])), 2),
            "canal_km_s": round(float(paso_canal * C_KMS / np.median(freq)), 3),
            "prob_falsa_alarma_por_linea": round(tasa_fa, 3) if np.isfinite(tasa_fa) else np.nan,
            "falsas_esperadas": round(tasa_fa * len(medidas), 2) if np.isfinite(tasa_fa) else np.nan,
            "prob_falsa_alarma_sin_catalogo": (round(tasa_fa_libre, 3)
                                               if np.isfinite(tasa_fa_libre) else np.nan),
            "prob_falsa_tentativa_sin_catalogo": (round(tasa_t_libre, 3)
                                                  if np.isfinite(tasa_t_libre) else np.nan),
            "falsas_esperadas_tentativas": (round(tasa_t_libre * len(medidas), 2)
                                            if np.isfinite(tasa_t_libre) else np.nan),
        })

        for m in medidas:
            inf = m["info"]
            catalogo_cols = {
                "E_up_K": round(float(inf["E_superior_K"]), 1) if inf is not None else np.nan,
                "log10_I": round(float(inf["log10_I_300K_nm2_MHz"]), 2) if inf is not None else np.nan,
                "A_ul_s-1": (float(f"{einstein_A(inf, Q300[molecula]):.3e}")
                             if inf is not None and molecula in Q300 else np.nan),
                "QN_superior": str(inf["QN_superior"]).strip() if inf is not None else "",
                "QN_inferior": str(inf["QN_inferior"]).strip() if inf is not None else "",
            }
            lineas_mol.append({
                "ventana": ventana, "molecula": molecula,
                "reposo_MHz": m["f0"],
                "observada_predicha_MHz": round(float(m["obs"]), 4),
                "V_linea_km_s": round(float(m["v1"]), 2) if np.isfinite(m["v1"]) else np.nan,
                "amplitud_Jy_beam": round(float(m["A"]), 4),
                "amplitud_sobre_ruido": round(float(m["snr"]), 1),
                "area_Jy_beam_km_s": round(float(m["W"]), 4),
                "error_area": round(float(m["sW"]), 4),
                **catalogo_cols,
                "area_esperada_LTE": round(float(m["W_lte"]), 4) if np.isfinite(m["W_lte"]) else np.nan,
                "desviacion_LTE": round(float(m["pull"]), 1) if np.isfinite(m["pull"]) else np.nan,
                "snr_integrado": round(float(m["snr_int"]), 1),
                "robusta_al_fondo": "sí" if m["robusta"] else "no",
                "robusta_tentativa": "sí" if m["robusta_t"] else "no",
                "sigma_km_s": round(float(m["s"] * C_KMS / m["obs"]), 3),
                "incertidumbre_MHz": round(float(m["inc"]), 4) if np.isfinite(m["inc"]) else np.nan,
                "p_valor_F": float(f"{m['p_f']:.2e}"),
                "estado": m["estado"],
                "motivo": m["motivo"],
            })
            bitacora.append({
                "molecula": molecula, "ventana": ventana,
                "frecuencia_reposo_MHz": m["f0"],
                "velocidad_molecula_km_s": round(v_med, 2),
                "velocidad_linea_km_s": round(float(m["v1"]), 2) if np.isfinite(m["v1"]) else np.nan,
                "frecuencia_observada_MHz": round(float(m["obs"]), 4),
                "amplitud_medida": round(float(m["A"]), 4),
                "snr": round(float(m["snr"]), 2),
                "area_Jy_beam_km_s": round(float(m["W"]), 4),
                **catalogo_cols,
                "criterio_snr_pico": "Aprobado" if m["snr"] >= SIGMA_DETECCION else "Rechazado",
                "criterio_snr_integrado": "Aprobado" if m["snr_int"] >= SNR_INTEGRADO_MIN else "Rechazado",
                "criterio_prueba_F": "Aprobado" if m["p_f"] < P_VALOR_F_MAX else "Rechazado",
                "criterio_velocidad": "Aprobado" if m["coherente"] else "Rechazado",
                "criterio_intensidades_LTE": ("No concluyente" if not lte_concluyente
                                              else "Aprobado" if m["lte_ok"] else "Rechazado"),
                "estado": m["estado"],
                "motivo": m["motivo"],
            })

        # ---- Gráfica: datos con huecos y modelo local de cada grupo ----
        xg, yg = con_huecos(x, y)
        ax.plot(xg, yg, label="Espectro observado", color="gray", alpha=0.7)
        for j, r in enumerate(pasada2):
            ax.plot(r["x"], r["mod"], color="crimson", lw=1.5,
                    label=f"Modelo local (V = {v_med:.2f} km/s)" if j == 0 else None)
            for c, a in zip(r["cen"], r["A"]):
                if a > 0:
                    ax.axvline(c, color="blue", linestyle="--", alpha=0.5)
            zoom.append((ventana, r, reposo[r["idx"]], v_med, medidas))
        ax.set_title(f"Molécula: {molecula} | Ventana: {ventana}")
        ax.set_xlabel("Frecuencia (MHz)")
        ax.set_ylabel("Intensidad (Jy/beam)")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle=":", alpha=0.6)

    if not filas:
        plt.close(fig)
        continue

    # ---- Estado de la molécula: transiciones independientes y corroboración entre ventanas ----
    con_det = [f for f in filas if f["transiciones_independientes"] > 0]
    n_ind = sum(f["transiciones_independientes"] for f in filas)
    if len(con_det) == 2:
        dif = abs(con_det[0]["V_LSR_km_s"] - con_det[1]["V_LSR_km_s"])
        concuerdan = ("sí" if dif <= CANALES_COHERENCIA * max(f["canal_km_s"] for f in con_det)
                      else "no")
        if concuerdan == "no":
            print(f"\nAVISO: W1 y W2 dan velocidades distintas para {molecula} "
                  f"({con_det[0]['V_LSR_km_s']} y {con_det[1]['V_LSR_km_s']} km/s).")
    else:
        concuerdan = "n/a"
    if n_ind >= MIN_TRANSICIONES:
        if len(con_det) == 2:
            estado_mol = "confirmada" if concuerdan == "sí" else "inconsistente"
        else:
            estado_mol = "probable"
    elif n_ind == 1:
        estado_mol = "candidata"
    else:
        estado_mol = "no detectada"
    for fila in filas:
        fila["ventanas_concuerdan"] = concuerdan
        fila["estado_sin_revisar_mezclas"] = estado_mol

    print(f"\n===== {molecula}: resultado por ventana =====")
    print(pd.DataFrame(filas).to_string(index=False))
    print(f"\n===== {molecula}: detalle de cada línea =====")
    print(pd.DataFrame(lineas_mol).to_string(index=False))

    plt.tight_layout()
    nombre_archivo_fig = f"evidencias_procesos/ajuste_{molecula}.png"
    plt.savefig(nombre_archivo_fig, dpi=300)
    print(f"Gráfica guardada en: {nombre_archivo_fig}")
    if automatico:
        plt.close(fig)
    else:
        plt.show()

    # ---- Recuadros por grupo de líneas (para ver el alineamiento de cada una) ----
    if zoom:
        if len(zoom) > MAX_PANELES_ZOOM:
            zoom = sorted(zoom, key=lambda z: -z[1]["A"].max())[:MAX_PANELES_ZOOM]
            zoom.sort(key=lambda z: (z[0], z[2][0]))
        cols = min(5, len(zoom))
        filas_z = int(np.ceil(len(zoom) / cols))
        figz, axz = plt.subplots(filas_z, cols, figsize=(3.6 * cols, 2.8 * filas_z),
                                 squeeze=False)
        for ax_z in axz.ravel():
            ax_z.set_visible(False)
        for ax_z, (vent, r, f0s, v_m, meds) in zip(axz.ravel(), zoom):
            ax_z.set_visible(True)
            ax_z.plot(r["x"], r["y"], color="gray", lw=0.9)
            ax_z.plot(r["x"], r["mod"], color="crimson", lw=1.3)
            for c in r["cen"]:
                ax_z.axvline(c, color="blue", linestyle="--", alpha=0.6)
            for f0 in f0s:
                ax_z.axvline(frecuencias_observadas(f0, v_m), color="black",
                             linestyle=":", alpha=0.6)
            snr_max = r["A"].max() / datos[vent][2]
            ax_z.set_title(f"{vent} {f0s[0]:.2f}  SNR {snr_max:.0f}", fontsize=8)
            ax_z.tick_params(labelsize=6)
        figz.suptitle(f"{molecula}: azul = centro ajustado, negro punteado = posición "
                      "esperada con la velocidad de la molécula", fontsize=9)
        plt.tight_layout()
        plt.savefig(f"evidencias_procesos/ajuste_zoom_{molecula}.png", dpi=200)
        plt.close(figz)

    resumen += filas
    detalle += lineas_mol


# ============== EXPORTACIÓN DE TABLAS DE JUSTIFICACIÓN ==============

if resumen:
    tabla = pd.DataFrame(resumen)
    print("\n===== RESUMEN DE TODAS LAS MOLÉCULAS =====")
    print(tabla.to_string(index=False))
    print(f"\nVelocidad mediana: {tabla['V_LSR_km_s'].median():.2f} km/s")

    detalle_df, tabla_mol = evaluar_mezclas_y_moleculas(detalle, resumen)
    estados_mol = dict(zip(tabla_mol["molecula"], tabla_mol["estado"]))
    print("\n===== ESTADO FINAL DE CADA MOLÉCULA (criterios de detección + mezclas) =====")
    print(tabla_mol.to_string(index=False))
    tabla_mol.to_csv("resumen_moleculas.csv", index=False)

    grafica_snr(detalle_df.to_dict("records"), estados_mol)

    clave = {(r["molecula"], r["ventana"], r["reposo_MHz"]): (r["mezcla_con"], r["estado_final"])
             for r in detalle_df.to_dict("records")}
    for b_ in bitacora:
        b_["mezcla_con"], b_["estado_final"] = clave.get(
            (b_["molecula"], b_["ventana"], b_["frecuencia_reposo_MHz"]), ("", ""))

    tabla.to_csv("ajuste_velocidad.csv", index=False)
    detalle_df.to_csv("lineas_ajustadas.csv", index=False)
    if bitacora:
        pd.DataFrame(bitacora).to_csv("evidencias_procesos/bitacora_sistema_doppler.csv", index=False)

    print("\n[Archivos de evidencia generados con éxito para tu justificación]:")
    print(" - ajuste_velocidad.csv")
    print(" - lineas_ajustadas.csv")
    print(" - resumen_moleculas.csv")
    print(" - evidencias_procesos/auditoria_ruido_espectros.csv")
    print(" - evidencias_procesos/bitacora_sistema_doppler.csv")
    print(" - evidencias_procesos/snr_lineas.png")
    print(" - Carpeta 'evidencias_procesos/' con gráficas de diagnóstico, ajuste y zoom por línea.")
else:
    print("\nNo se obtuvo ningún ajuste.")