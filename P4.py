from PIL import Image
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Funciones ya dadas para trabajar en la implementacion de 16QAM

def fuente_info(imagen):
    '''Una función que simula una fuente de
    información al importar una imagen y 
    retornar un vector de NumPy con las 
    dimensiones de la imagen, incluidos los
    canales RGB: alto x largo x 3 canales

    :param imagen: Una imagen en formato JPG
    :return: un vector de pixeles
    '''
    img = Image.open(imagen)
    
    return np.array(img)


def rgb_a_bit(array_imagen):
    '''Convierte los pixeles de base
    decimal (de 0 a 255) a binaria
    (de 00000000 a 11111111).

    :param imagen: array de una imagen
    :return: Un vector de (1 x k) bits 'int'
    '''
    # Obtener las dimensiones de la imagen
    x, y, z = array_imagen.shape

    # Número total de elementos (pixeles x canales)
    n_elementos = x * y * z

    # Convertir la imagen a un vector unidimensional de n_elementos
    pixeles = np.reshape(array_imagen, n_elementos)

    # Convertir los canales a base 2
    bits = [format(pixel, '08b') for pixel in pixeles]
    bits_Rx = np.array(list(''.join(bits)))

    return bits_Rx.astype(int)

def modulador(bits, fc, mpp):
    '''Un método que simula el esquema de
    modulación digital BPSK.

    :param bits: Vector unidimensional de bits
    :param fc: Frecuencia de la portadora en Hz
    :param mpp: Cantidad de muestras por periodo de onda portadora
    :return: Un vector con la señal modulada
    :return: Un valor con la potencia promedio [W]
    :return: La onda portadora c(t)
    :return: La onda cuadrada moduladora (información)
    '''
    # 1. Parámetros de la 'señal' de información (bits)
    N = len(bits) # Cantidad de bits

    # 2. Construyendo un periodo de la señal portadora c(t)
    Tc = 1 / fc  # periodo [s]
    t_periodo = np.linspace(0, Tc, mpp)  # mpp: muestras por período
    portadora = np.sin(2*np.pi*fc*t_periodo)

    # 3. Inicializar la señal modulada s(t)
    t_simulacion = np.linspace(0, N*Tc, N*mpp)
    senal_Tx = np.zeros(t_simulacion.shape)
    moduladora = np.zeros(t_simulacion.shape)  # (opcional) señal de bits

    # 4. Asignar las formas de onda según los bits (BPSK)
    for i, bit in enumerate(bits):
        if bit == 1:
            senal_Tx[i*mpp : (i+1)*mpp] = portadora
            moduladora[i*mpp : (i+1)*mpp] = 1
        else:
            senal_Tx[i*mpp : (i+1)*mpp] = portadora * -1
            moduladora[i*mpp : (i+1)*mpp] = 0

    # 5. Calcular la potencia promedio de la señal modulada
    P_senal_Tx = (1 / (N*Tc)) * np.trapz(pow(senal_Tx, 2), t_simulacion)

    return senal_Tx, P_senal_Tx, portadora, moduladora

def canal_ruidoso(senal_Tx, Pm, SNR):
    '''Un bloque que simula un medio de trans-
    misión no ideal (ruidoso) empleando ruido
    AWGN. Pide por parámetro un vector con la
    señal provieniente de un modulador y un
    valor en decibelios para la relación señal
    a ruido.

    :param senal_Tx: El vector del modulador
    :param Pm: Potencia de la señal modulada
    :param SNR: Relación señal-a-ruido en dB
    :return: La señal modulada al dejar el canal
    '''
    # Potencia del ruido generado por el canal
    Pn = Pm / pow(10, SNR/10)

    # Generando ruido auditivo blanco gaussiano (potencia = varianza)
    ruido = np.random.normal(0, np.sqrt(Pn), senal_Tx.shape)

    # Señal distorsionada por el canal ruidoso
    senal_Rx = senal_Tx + ruido

    return senal_Rx

def demodulador(senal_Rx, portadora, mpp):
    '''Un método que simula un bloque demodulador
    de señales, bajo un esquema BPSK. El criterio
    de demodulación se basa en decodificación por
    detección de energía.

    :param senal_Rx: La señal recibida del canal
    :param portadora: La onda portadora c(t)
    :param mpp: Número de muestras por periodo
    :return: Los bits de la señal demodulada
    '''
    # Cantidad de muestras en senal_Rx
    M = len(senal_Rx)

    # Cantidad de bits (símbolos) en transmisión
    N = int(M / mpp)

    # Vector para bits obtenidos por la demodulación
    bits_Rx = np.zeros(N)

    # Vector para la señal demodulada
    senal_demodulada = np.zeros(senal_Rx.shape)

    # Pseudo-energía de un período de la portadora
    Es = np.sum(portadora * portadora)

    # Demodulación
    for i in range(N):
        # Producto interno de dos funciones
        producto = senal_Rx[i*mpp : (i+1)*mpp] * portadora
        Ep = np.sum(producto)
        senal_demodulada[i*mpp : (i+1)*mpp] = producto

        # Criterio de decisión por detección de energía
        if Ep > 0:
            bits_Rx[i] = 1
        else:
            bits_Rx[i] = 0

    return bits_Rx.astype(int), senal_demodulada

def bits_a_rgb(bits_Rx, dimensiones):
    '''Un blque que decodifica el los bits
    recuperados en el proceso de demodulación

    :param: Un vector de bits 1 x k 
    :param dimensiones: Tupla con dimensiones de la img.
    :return: Un array con los pixeles reconstruidos
    '''
    # Cantidad de bits
    N = len(bits_Rx)

    # Se reconstruyen los canales RGB
    bits = np.split(bits_Rx, N / 8)

    # Se decofican los canales:
    canales = [int(''.join(map(str, canal)), 2) for canal in bits]
    pixeles = np.reshape(canales, dimensiones)

    return pixeles.astype(np.uint8)



# Funciones para modulación 16QAM implementadas
# parte 4.1
# Se implementa modulador 16QAM
def modulador_16QAM(bits, fc, mpp):
    '''
    :param bits: Vector unidimensional de bits
    :param fc: Frecuencia de la portadora en Hz
    :param mpp: Cantidad de muestras por periodo de onda portadora
    :return: Un vector con la señal modulada
    :return: Un valor con la potencia promedio [W]
    :return: La onda portadora c(t)
    :return: La onda cuadrada moduladora (información)
    '''
    # 1. Parámetros de la 'señal' de información (bits)
    N = len(bits) # Cantidad bits


    # 2. Construyendo un periodo de la señal portadora c(t)
    Tc = 1 / fc  # periodo [s]
    t_periodo = np.linspace(0, Tc, mpp)
    portadora_cos = np.cos(2*np.pi*fc*t_periodo)
    portadora_sen = np.sin(2*np.pi*fc*t_periodo)


    # cantidad de símbolos de 4 bits en el flujo de bits
    bits_chunk = int(len(bits)/4)
    print("cantidad de símbolos en array de bits")
    print(bits_chunk)

    # 3. Inicializar la señal modulada s(t)
    t_simulacion = np.linspace(0, N*Tc, N*mpp)
    senal_Tx = np.zeros(t_simulacion.shape)
    moduladora_cos = np.zeros(t_simulacion.shape)  # señal de información
    moduladora_sin = np.zeros(t_simulacion.shape)  # señal de información

    # 4. Asignar las formas de onda según los bits 16_QAM
    # Preparación de datos para pasar a las portadores seno y coseno
    for i in range(0, len(bits), 4):
        if (bits[i] == 0) and (bits[i+1] == 0) and (bits[i+2] == 0) and (bits[i+2] == 0):
            senal_Tx[i*mpp : (i+1)*mpp] = portadora_cos * -3
            senal_Tx[(i+1)*mpp : (i+2)*mpp] = portadora_cos * -3
            senal_Tx[(i+2)*mpp : (i+3)*mpp] = portadora_sen * 3
            senal_Tx[(i+3)*mpp : (i+4)*mpp] = portadora_sen * 3
            moduladora[i*mpp : (i+1)*mpp] = 0
            moduladora[(i+1)*mpp : (i+2)*mpp] = 0
            moduladora[(i+2)*mpp : (i+3)*mpp] = 0
            moduladora[(i+3)*mpp : (i+4)*mpp] = 0
        elif (bits[i] == 0) and (bits[i+1] == 0) and (bits[i+2] == 0) and (bits[i+2] == 1):
            senal_Tx[i*mpp : (i+1)*mpp] = portadora_cos * -3
            senal_Tx[(i+1)*mpp : (i+2)*mpp] = portadora_cos * -3
            senal_Tx[(i+2)*mpp : (i+3)*mpp] = portadora_sen * 1
            senal_Tx[(i+3)*mpp : (i+4)*mpp] = portadora_sen * 1
            moduladora[i*mpp : (i+1)*mpp] = 0
            moduladora[(i+1)*mpp : (i+2)*mpp] = 0
            moduladora[(i+2)*mpp : (i+3)*mpp] = 0
            moduladora[(i+3)*mpp : (i+4)*mpp] = 1
        elif (bits[i] == 0) and (bits[i+1] == 0) and (bits[i+2] == 1) and (bits[i+2] == 0):
            senal_Tx[i*mpp : (i+1)*mpp] = portadora_cos * -3
            senal_Tx[(i+1)*mpp : (i+2)*mpp] = portadora_cos * -3
            senal_Tx[(i+2)*mpp : (i+3)*mpp] = portadora_sen * -3
            senal_Tx[(i+3)*mpp : (i+4)*mpp] = portadora_sen * -3
            moduladora[i*mpp : (i+1)*mpp] = 0
            moduladora[(i+1)*mpp : (i+2)*mpp] = 0
            moduladora[(i+2)*mpp : (i+3)*mpp] = 1
            moduladora[(i+3)*mpp : (i+4)*mpp] = 0
        elif (bits[i] == 0) and (bits[i+1] == 0) and (bits[i+2] == 1) and (bits[i+2] == 1):
            senal_Tx[i*mpp : (i+1)*mpp] = portadora_cos * -3
            senal_Tx[(i+1)*mpp : (i+2)*mpp] = portadora_cos * -3
            senal_Tx[(i+2)*mpp : (i+3)*mpp] = portadora_sen * -1
            senal_Tx[(i+3)*mpp : (i+4)*mpp] = portadora_sen * -1
            moduladora[i*mpp : (i+1)*mpp] = 0
            moduladora[(i+1)*mpp : (i+2)*mpp] = 0
            moduladora[(i+2)*mpp : (i+3)*mpp] = 1
            moduladora[(i+3)*mpp : (i+4)*mpp] = 1
        elif (bits[i] == 0) and (bits[i+1] == 1) and (bits[i+2] == 0) and (bits[i+2] == 0):
            senal_Tx[i*mpp : (i+1)*mpp] = portadora_cos * -1
            senal_Tx[(i+1)*mpp : (i+2)*mpp] = portadora_cos * -1
            senal_Tx[(i+2)*mpp : (i+3)*mpp] = portadora_sen * 3
            senal_Tx[(i+3)*mpp : (i+4)*mpp] = portadora_sen * 3
            moduladora[i*mpp : (i+1)*mpp] = 0
            moduladora[(i+1)*mpp : (i+2)*mpp] = 0
            moduladora[(i+2)*mpp : (i+3)*mpp] = 1
            moduladora[(i+3)*mpp : (i+4)*mpp] = 0
        elif (bits[i] == 0) and (bits[i+1] == 1) and (bits[i+2] == 0) and (bits[i+2] == 1):
            senal_Tx[i*mpp : (i+1)*mpp] = portadora_cos * -1
            senal_Tx[(i+1)*mpp : (i+2)*mpp] = portadora_cos * -1
            senal_Tx[(i+2)*mpp : (i+3)*mpp] = portadora_sen * 1
            senal_Tx[(i+3)*mpp : (i+4)*mpp] = portadora_sen * 1
            moduladora[i*mpp : (i+1)*mpp] = 0
            moduladora[(i+1)*mpp : (i+2)*mpp] = 1
            moduladora[(i+2)*mpp : (i+3)*mpp] = 0
            moduladora[(i+3)*mpp : (i+4)*mpp] = 1
        elif (bits[i] == 0) and (bits[i+1] == 1) and (bits[i+2] == 1) and (bits[i+2] == 0):
            senal_Tx[i*mpp : (i+1)*mpp] = portadora_cos * -1
            senal_Tx[(i+1)*mpp : (i+2)*mpp] = portadora_cos * -1
            senal_Tx[(i+2)*mpp : (i+3)*mpp] = portadora_sen * -3
            senal_Tx[(i+3)*mpp : (i+4)*mpp] = portadora_sen * -3
            moduladora[i*mpp : (i+1)*mpp] = 0
            moduladora[(i+1)*mpp : (i+2)*mpp] = 1
            moduladora[(i+2)*mpp : (i+3)*mpp] = 1
            moduladora[(i+3)*mpp : (i+4)*mpp] = 0
        elif (bits[i] == 0) and (bits[i+1] == 1) and (bits[i+2] == 1) and (bits[i+2] == 1):
            senal_Tx[i*mpp : (i+1)*mpp] = portadora_cos * -1
            senal_Tx[(i+1)*mpp : (i+2)*mpp] = portadora_cos * -1
            senal_Tx[(i+2)*mpp : (i+3)*mpp] = portadora_sen * -1
            senal_Tx[(i+3)*mpp : (i+4)*mpp] = portadora_sen * -1
            moduladora[i*mpp : (i+1)*mpp] = 0
            moduladora[(i+1)*mpp : (i+2)*mpp] = 1
            moduladora[(i+2)*mpp : (i+3)*mpp] = 1
            moduladora[(i+3)*mpp : (i+4)*mpp] = 1
        elif (bits[i] == 1) and (bits[i+1] == 0) and (bits[i+2] == 0) and (bits[i+2] == 0):
            senal_Tx[i*mpp : (i+1)*mpp] = portadora_cos * 3
            senal_Tx[(i+1)*mpp : (i+2)*mpp] = portadora_cos * 3
            senal_Tx[(i+2)*mpp : (i+3)*mpp] = portadora_sen * 3
            senal_Tx[(i+3)*mpp : (i+4)*mpp] = portadora_sen * 3
            moduladora[i*mpp : (i+1)*mpp] = 1
            moduladora[(i+1)*mpp : (i+2)*mpp] = 0
            moduladora[(i+2)*mpp : (i+3)*mpp] = 0
            moduladora[(i+3)*mpp : (i+4)*mpp] = 0
        elif (bits[i] == 1) and (bits[i+1] == 0) and (bits[i+2] == 0) and (bits[i+2] == 1):
            senal_Tx[i*mpp : (i+1)*mpp] = portadora_cos * 3
            senal_Tx[(i+1)*mpp : (i+2)*mpp] = portadora_cos * 3
            senal_Tx[(i+2)*mpp : (i+3)*mpp] = portadora_sen * 1
            senal_Tx[(i+3)*mpp : (i+4)*mpp] = portadora_sen * 1
            moduladora[i*mpp : (i+1)*mpp] = 1
            moduladora[(i+1)*mpp : (i+2)*mpp] = 0
            moduladora[(i+2)*mpp : (i+3)*mpp] = 0
            moduladora[(i+3)*mpp : (i+4)*mpp] = 1
        elif (bits[i] == 1) and (bits[i+1] == 0) and (bits[i+2] == 1) and (bits[i+2] == 0):
            senal_Tx[i*mpp : (i+1)*mpp] = portadora_cos * 3
            senal_Tx[(i+1)*mpp : (i+2)*mpp] = portadora_cos * 3
            senal_Tx[(i+2)*mpp : (i+3)*mpp] = portadora_sen * -3
            senal_Tx[(i+3)*mpp : (i+4)*mpp] = portadora_sen * -3
            moduladora[i*mpp : (i+1)*mpp] = 1
            moduladora[(i+1)*mpp : (i+2)*mpp] = 0
            moduladora[(i+2)*mpp : (i+3)*mpp] = 1
            moduladora[(i+3)*mpp : (i+4)*mpp] = 0
        elif (bits[i] == 1) and (bits[i+1] == 0) and (bits[i+2] == 1) and (bits[i+2] == 1):
            senal_Tx[i*mpp : (i+1)*mpp] = portadora_cos * 3
            senal_Tx[(i+1)*mpp : (i+2)*mpp] = portadora_cos * 3
            senal_Tx[(i+2)*mpp : (i+3)*mpp] = portadora_sen * -1
            senal_Tx[(i+3)*mpp : (i+4)*mpp] = portadora_sen * -1
            moduladora[i*mpp : (i+1)*mpp] = 1
            moduladora[(i+1)*mpp : (i+2)*mpp] = 0
            moduladora[(i+2)*mpp : (i+3)*mpp] = 1
            moduladora[(i+3)*mpp : (i+4)*mpp] = 1
        elif (bits[i] == 1) and (bits[i+1] == 1) and (bits[i+2] == 0) and (bits[i+2] == 0):
            senal_Tx[i*mpp : (i+1)*mpp] = portadora_cos * 1
            senal_Tx[(i+1)*mpp : (i+2)*mpp] = portadora_cos * 1
            senal_Tx[(i+2)*mpp : (i+3)*mpp] = portadora_sen * 3
            senal_Tx[(i+3)*mpp : (i+4)*mpp] = portadora_sen * 3
            moduladora[i*mpp : (i+1)*mpp] = 1
            moduladora[(i+1)*mpp : (i+2)*mpp] = 1
            moduladora[(i+2)*mpp : (i+3)*mpp] = 0
            moduladora[(i+3)*mpp : (i+4)*mpp] = 0
        elif (bits[i] == 1) and (bits[i+1] == 1) and (bits[i+2] == 0) and (bits[i+2] == 1):
            senal_Tx[i*mpp : (i+1)*mpp] = portadora_cos * 1
            senal_Tx[(i+1)*mpp : (i+2)*mpp] = portadora_cos * 1
            senal_Tx[(i+2)*mpp : (i+3)*mpp] = portadora_sen * 1
            senal_Tx[(i+3)*mpp : (i+4)*mpp] = portadora_sen * 1
            moduladora[i*mpp : (i+1)*mpp] = 1
            moduladora[(i+1)*mpp : (i+2)*mpp] = 1
            moduladora[(i+2)*mpp : (i+3)*mpp] = 0
            moduladora[(i+3)*mpp : (i+4)*mpp] = 1
        elif (bits[i] == 1) and (bits[i+1] == 1) and (bits[i+2] == 1) and (bits[i+2] == 0):
            senal_Tx[i*mpp : (i+1)*mpp] = portadora_cos * 1
            senal_Tx[(i+1)*mpp : (i+2)*mpp] = portadora_cos * 1
            senal_Tx[(i+2)*mpp : (i+3)*mpp] = portadora_sen * -3
            senal_Tx[(i+3)*mpp : (i+4)*mpp] = portadora_sen * -3
            moduladora[i*mpp : (i+1)*mpp] = 1
            moduladora[(i+1)*mpp : (i+2)*mpp] = 1
            moduladora[(i+2)*mpp : (i+3)*mpp] = 1
            moduladora[(i+3)*mpp : (i+4)*mpp] = 0
        else:
            senal_Tx[i*mpp : (i+1)*mpp] = portadora_cos * 1
            senal_Tx[(i+1)*mpp : (i+2)*mpp] = portadora_cos * 1
            senal_Tx[(i+2)*mpp : (i+3)*mpp] = portadora_sen * -1
            senal_Tx[(i+3)*mpp : (i+4)*mpp] = portadora_sen * -1
            moduladora[i*mpp : (i+1)*mpp] = 1
            moduladora[(i+1)*mpp : (i+2)*mpp] = 1
            moduladora[(i+2)*mpp : (i+3)*mpp] = 1
            moduladora[(i+3)*mpp : (i+4)*mpp] = 1
    # 5. Calcular la potencia promedio de la señal modulada
    Pm_senal_Tx = (1 / (N*Tc)) * np.trapz(pow(senal_Tx, 2), t_simulacion)


    # Se combinan las señales
    portadora_16QAM = portadora_cos + portadora_sen

    # 5. Calcular la potencia promedio de la señal modulada

    return senal_Tx, Pm_senal_Tx, portadora_16QAM, moduladora


def demodulador_16QAM(senal_Rx, portadora_16QAM, mpp):
    '''Un método que simula un bloque demodulador
    de señales, bajo un esquema 16QAM. El criterio
    de demodulación se basa en decodificación por
    detección de energía.

    :param senal_Rx: La señal recibida del canal
    :param portadora_sen: La onda portadora c(t)
    :param mpp: Número de muestras por periodo
    :return: Los bits de la señal demodulada
    '''
    # Cantidad de muestras en senal_Rx
    M = len(senal_Rx)

    # Cantidad de símbolos en transmisión
    bits_chunk = int(M / mpp)

    # Vector de bits obtenidos para demodulación, 4 bits por símbolo
    bits_Rx = np.zeros(bits_chunk*4)

    # Vector para la señal demodulada
    senal_demodulada = np.zeros(senal_Rx.shape)

    # Pseudo-energía de un período de la portadora
    Es = np.sum(portadora_16QAM**2)

    # Demodulación
    for i in range(N):
        # Producto interno de dos funciones
        producto = senal_Rx[i*mpp : (i+1)*mpp] * portadora_16QAM
        Ep = np.sum(producto)
        senal_demodulada[i*mpp : (i+1)*mpp] = producto

        # Criterio de decisión por detección de energía
        if Ep > 0:
            bits_Rx[i] = 1
        else:
            bits_Rx[i] = 0
    return bits_Rx.astype(int), senal_demodulada



# Simulación de modulación y demodulación

import matplotlib.pyplot as plt
import time

# Parámetros
fc = 5000  # frecuencia de la portadora
mpp = 20   # muestras por periodo de la portadora
SNR = -5    # relación señal-a-ruido del canal

# Iniciar medición del tiempo de simulación
inicio = time.time()

# 1. Importar y convertir la imagen a trasmitir
imagen_TX = fuente_info('arenal.jpg')
dimensiones_16QAM = imagen_TX.shape

# 2. Codificar los pixeles de la imagen
bits_TX = rgb_a_bit(imagen_TX)

# 3. Modular la cadena de bits usando el esquema 16QAM
# Se implementa la función moduladora para obtener las dos partes de la señal
senal_TX, Pm_senal_TX, portadora_16QAM, moduladora_16QAM= modulador_16QAM(bits_TX, fc, mpp)



# Se suman las señales sen y cos para  transmite la señal
# modulada completa  por un canal ruidoso

senal_RX = canal_ruidoso(senal_TX, Pm_16QAM, SNR)

# 5. Se desmodula la señal recibida del canal
# Se suman las señales
portadora_16QAM = portadora_cos + portadora_sen
bits_RX, senal_demodulada_16QAM = demodulador(senal_RX, portadora_16QAM, mpp)

# 6. Se visualiza la imagen recibida 
imagen_RX = bits_a_rgb(bits_RX, dimensiones_16QAM)
Fig = plt.figure(figsize=(10,6))

# Cálculo del tiempo de simulación
print('Duración de la simulación: ', time.time() - inicio)

# 7. Calcular número de errores
errores_16QAM = sum(abs(bits_TX - bits_RX))
BER_16QAM = errores_16QAM/len(bits_TX)
print('{} errores, para un BER de {:0.4f}.'.format(errores_16QAM, BER_16QAM))

# Mostrar imagen transmitida
ax = Fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(imagen_Tx)
ax.set_title('Transmitido')

# Mostrar imagen recuperada
ax = Fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(imagen_RX)
ax.set_title('Recuperado')
Fig.tight_layout()

plt.imshow(imagen_RX)


# Visualizar el cambio entre las señales
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, sharex=True, figsize=(14, 7))

# La onda cuadrada moduladora (bits de entrada)
ax1.plot(moduladora[0:600], color='r', lw=2)
ax1.set_ylabel('$b(t)$')

# La señal modulada por BPSK
ax2.plot(senal_Tx[0:600], color='g', lw=2)
ax2.set_ylabel('$s(t)$')

# La señal modulada al dejar el canal
ax3.plot(senal_Rx[0:600], color='b', lw=2)
ax3.set_ylabel('$s(t) + n(t)$')

# La señal demodulada
ax4.plot(senal_demodulada[0:600], color='m', lw=2)
ax4.set_ylabel('$b^{\prime}(t)$')
ax4.set_xlabel('$t$ / milisegundos')
fig.tight_layout()
plt.show()


# gráfica de valor esperado
# Creación del vector de tiempo
T = 200			# número de elementos (usamos solo 200, no el total de 600)
t_final = 600	# tiempo en segundos
t = np.linspace(0, t_final, T)

P = [np.mean(senal_TX[i]) for i in range(200)]
plt.xlabel('Realizaciones')
plt.ylabel('Valor esperado')
plt.plot(P, lw=2)

Prom = np.mean(P, dtype=int)  #Se convierte en entero, ya que la función mean devuelve un float
print("El valor promedio es: ", Prom)



