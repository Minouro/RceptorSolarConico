import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, RadioButtons, TextBox
from matplotlib import gridspec
import matplotlib as mpl
from matplotlib.patches import Circle
import pandas as pd
import seaborn as sns

df = pd.read_csv('ar.txt', sep='\t', header=None, names=['Temperatura', 'Densidade', 'Condutividade Termica', 'Calor Especifico', 'Viscosidade Dinamica', 'Viscosidade Cinematica', 'Pr'], index_col=None)

def reynolds(densidade, velocidade, diametro, viscosidade):
    return (diametro*densidade*velocidade)/viscosidade

def transf_coef(nusselt, condutividade, diametro):
    return (nusselt * condutividade) / diametro

def gnielinski(re, pr):
    f = (0.782*np.log(re)-1.5)**-2
    return ((f/8)*(re-1000)*pr)/(1+12.7*(f/8)**(1/2)*(pr**(2/3)-1))

def nusselt_relation(reynolds, prandtl):
    return 0.023 * (reynolds ** 0.8) * (prandtl ** 0.4)

def nusselt_externo(re, pr, C, m):
    return C*re**m*pr**(1/3)

def loss_convection(comprimento, diametro_ext, coeficiente, temp_ext, temp_amb=20+273.15):
    return comprimento * np.pi * diametro_ext * coeficiente * (temp_ext - temp_amb)

def loss_radiation(emissividade, diametro_ext, comprimento, temp_ext, temp_amb = 20+273.15, const_radiacao=5.669 * 10**-8):
    return emissividade * const_radiacao * np.pi * diametro_ext * comprimento * ((temp_ext ** 4) - (temp_amb ** 4))

def propriedades(temp_entrada):
    resultado = df.loc[df['Temperatura'] >= temp_entrada]

    if resultado.empty:
        raise ValueError(f"Temperatura de entrada ({temp_entrada}) está fora do intervalo das temperaturas disponíveis.")

    return resultado.iloc[0]

def velocidade_int(calor_util, densidade, area, calor_especifico, tempE, tempS):
    return calor_util/(densidade*area*calor_especifico*(tempS-tempE))

def temp_int(Q, area, h, temp_m):
    return Q/(area*h) + temp_m

def temp_ext(Q, d_ext, d_int, comprimento, condutividade, temp_int):
    return (Q*np.log(d_ext/d_int))/(2*np.pi*comprimento*condutividade) + temp_int

def solar(diametro_interno = 0.05):
    #-----------------------------------------Interno----------------------------------------------------------
    temp_entrada = float(text_tempE.text) + 273.15
    temp_saida = float(text_tempS.text) + 273.15
    calor_util = float(text_calor.text)
    radiacao = float(text_radiacao.text)
    diametro_externo = diametro_interno + prop['espessura']
    refletividade = prop['refletividade']
    comprimento = prop['comprimento']

    temp_media = (temp_entrada + temp_saida) / 2
    propriedades_ar = propriedades(temp_media-273.15)

    densidade = propriedades_ar['Densidade']
    viscosidade = propriedades_ar['Viscosidade Dinamica']
    condutividade = propriedades_ar['Condutividade Termica']
    calor_especifico = propriedades_ar['Calor Especifico']
    pr = propriedades_ar['Pr']
    
    area_int = np.pi*(diametro_interno/2)**2
    area_sup = np.pi*diametro_interno*comprimento
    
    velocidade = velocidade_int(calor_util, densidade, area_int, calor_especifico, temp_entrada, temp_saida)
    re = reynolds(densidade, velocidade, diametro_interno, viscosidade)

    if re < 2300:
        nu = 4.36
        turbulento = "Laminar"
    elif re <= 10000:
        nu = gnielinski(re, pr)
        turbulento = "Meio-Turbulento"
    else:
        nu = nusselt_relation(re, pr)
        turbulento = "Turbulento"

    coef_transferencia = transf_coef(nu, condutividade, diametro_interno)
    temp_interna = temp_int(calor_util, area_sup, coef_transferencia, temp_media)

    #-----------------------------------------Externo----------------------------------------------------------

    propriedade_ambiente = propriedades(20)
    densidade_ambiente = propriedade_ambiente['Densidade']
    viscosidade_ambiente = propriedade_ambiente['Viscosidade Dinamica']
    condutividade_ambiente = propriedade_ambiente['Condutividade Termica']
    pr_ambiente = propriedade_ambiente['Pr']
    
    re_ambiente = reynolds(densidade_ambiente, 0.75, diametro_externo, viscosidade_ambiente)
    
    if re_ambiente < 4:
        c = 0.989
        m = 0.330
    elif re_ambiente < 40:
        c = 0.911
        m = 0.385
    elif re_ambiente < 4000:
        c = 0.683
        m = 0.466
    elif re_ambiente < 40000:
        c = 0.193
        m = 0.618
    else:
        c = 0.027
        m = 0.805
        
    nu_ambiente = nusselt_externo(re_ambiente, pr_ambiente, c, m)
    coef_transf_ambiente = transf_coef(nu_ambiente, condutividade_ambiente, diametro_externo)

    temp_externa = temp_ext(calor_util, diametro_externo, diametro_interno, comprimento, condutividade_material, temp_interna)

    perda_conv = loss_convection(comprimento, diametro_externo, coef_transf_ambiente, temp_externa)
    perda_rad = loss_radiation(emissividade_material, diametro_externo, comprimento, temp_externa)

    calor_incidente = calor_util + perda_conv + perda_rad
    area_heliostato = calor_incidente/(radiacao*refletividade*absortividade_material)

    return velocidade, temp_interna, temp_externa, perda_conv, perda_rad, calor_incidente, area_heliostato, re, turbulento, nu
#-------------------------------------------------------------------------------------------------------------

def helice_conica(d_tubo, d_base, d_topo, altura, num_pontos=1000):
    voltas = round(altura/d_tubo)

    z = np.linspace(0, altura, voltas * num_pontos)
    raio = np.linspace(d_base / 2, d_topo / 2, len(z))
    theta = np.linspace(0, 2 * np.pi * voltas, len(z))  # Repete 2π vezes o número de voltas
    x = raio * np.cos(theta)
    y = raio * np.sin(theta)

    return x, y, z, voltas

def calcular_comprimento_helice(x, y, z):
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)
    comprimento = np.sum(np.sqrt(dx**2 + dy**2 + dz**2))
    return comprimento

def plotar_helice(ax, x, y, z, d):
    for coll in ax.collections:
        coll.remove()

    if prop['d_base'] >= 0.4 or prop['d_topo'] >= 0.4 or prop['altura'] >= 0.4:
        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-0.5, 0.5])
        ax.set_zlim([-0.5, 0.5])
        multiplicador = 50
    elif prop['d_base'] >= 0.3 or prop['d_topo'] >= 0.3 or prop['altura'] >= 0.3:
        ax.set_xlim([-0.3, 0.3])
        ax.set_ylim([-0.3, 0.3])
        ax.set_zlim([-0.3, 0.3])
        multiplicador = 300
    elif prop['d_base'] >= 0.1 or prop['d_topo'] >= 0.1 or prop['altura'] >= 0.1:
        ax.set_xlim([-0.1, 0.1])
        ax.set_ylim([-0.1, 0.1])
        ax.set_zlim([-0.1, 0.1])
        multiplicador = 2000

    ax.scatter(x, y, z, c=np.linspace(0.1, 100, len(x)), cmap=mpl.colormaps['plasma'], s=d*multiplicador)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')   
    plt.draw()

def plotar_tubo(ax, diametro_interno, diametro_externo):
    ax.clear()
    ax.text(0.05, 0.82, f"Diâmetro Interno = {diametro_interno:.4f} m", transform=ax.transAxes)
    ax.text(0.05, 0.90, f"Diâmetro Externo = {diametro_externo:.4f} m", transform=ax.transAxes)
    ax.text(0.65, 0.02, f"k = {condutividade_material:.2f}", transform=ax.transAxes)
    ax.text(0.05, 0.02, f"α = {absortividade_material:.2f}", transform=ax.transAxes)	
    ax.text(0.35, 0.02, f"ε = {emissividade_material:.2f}", transform=ax.transAxes)

    ax.add_patch(Circle((0, 0), diametro_externo, edgecolor='black', facecolor='none', lw=2, hatch='////'))
    ax.add_patch(Circle((0, 0), diametro_interno, edgecolor='black', facecolor='white', lw=2))
    
    ax.set_aspect('equal')  # Manter a proporção dos eixos iguais
    ax.set_xlim([-diametro_externo*2, diametro_externo*2])
    ax.set_ylim([-diametro_externo*2, diametro_externo*2])
    ax.grid(color='gray', alpha=0.2)
    plt.draw()

def graph(ax, x, y, color, label_x, label_y, atual_x, atual_y):
    ax.cla()
    ax.plot(x, y, color=color)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.plot(atual_x, atual_y, 'ro')
    ax.axvline(atual_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(atual_y, color='gray', linestyle='--', alpha=0.5)
    fig_graph.canvas.draw()

def atualiza_grafico():
    x, y, z, voltas = helice_conica(prop['d_tubo'], prop['d_base'], prop['d_topo'], prop['altura'])
    prop['comprimento'] = calcular_comprimento_helice(x, y, z)
    prop['voltas'] = voltas
    L_text.set_text(f"Comprimento = {prop['comprimento']:.2f} m")
    voltas_text.set_text(f"Voltas = {prop['voltas']}")
    
    velocidade, temp_interna, temp_externa, perda_conv, perda_rad, calor_incidente, area_heliostato, re, turbulento, nu  = solar(prop['d_tubo'])
    solar_text.set_text(f"Taxa de Calor Incidente = {calor_incidente:.4f} W")
    area_text.set_text(f"Área Total dos Heliostatos = {area_heliostato:.4f} m\u00b2")
    
    plotar_helice(ax_helice, x, y, z, prop['d_tubo'])
    plotar_tubo(ax_tubo, prop['d_tubo'], prop['d_tubo'] + prop['espessura'])
    atualizar_grafico_solar(velocidade, perda_conv, perda_rad, calor_incidente, temp_externa, temp_interna, re, area_heliostato, turbulento, nu)

def atualizar_grafico_solar(velocidade, perda_conv, perda_rad, calor_incidente, temp_ext, temp_int, reynolds, area, turbulento, nu):
    ax_solar.clear()

    ax_solar.text(0.01, 0.86, f"Perda por Convecção = {perda_conv:.4f} W", transform=ax_solar.transAxes)
    ax_solar.text(0.01, 0.72, f"Perda por Radiação = {perda_rad:.4f} W", transform=ax_solar.transAxes)
    ax_solar.text(0.01, 0.58, f"Reynolds = {reynolds:.4f} ({turbulento})", transform=ax_solar.transAxes)
    ax_solar.text(0.60, 0.38, f"Temp. Superfície Externa = {(temp_ext - 273.15):.4f} °C", transform=ax_solar.transAxes)
    ax_solar.text(0.60, 0.24, f"Temp. Superfície Interna = {(temp_int - 273.15):.4f} °C", transform=ax_solar.transAxes)
    ax_solar.text(0.60, 0.10, f"Velocidade de Entrada do Ar = {velocidade:.4f} m/s", transform=ax_solar.transAxes)

    atualizado = np.linspace(0.01, 0.1, 100)
    solar_atualizado = np.vectorize(lambda n: solar(n)[5] / 1e3)(atualizado)

    ax_solar.plot(atualizado, solar_atualizado, color='orange')

    solar_atual = calor_incidente / 1e3
    ax_solar.plot(prop['d_tubo'], solar_atual, 'ro')
    ax_solar.axvline(prop['d_tubo'], color='gray', linestyle='--', alpha=0.5)
    ax_solar.axhline(solar_atual, color='gray', linestyle='--', alpha=0.5)
    ax_solar.set_xlim([0.01, 0.1])
    ax_solar.set_xticks(np.linspace(0.01,0.1,11))
    ax_solar.set_xlabel('Diâmetro do Tubo (m)')
    ax_solar.set_ylabel('Taxa de Calor Incidente (kW)')
    ax_solar.grid(color='gray', alpha=0.5)

    intervalo = np.linspace(0.01, 0.1, 100)
    reynolds_atualizado = np.vectorize(lambda n: solar(n)[7])(intervalo)
    velocidade_atualizado = np.vectorize(lambda n: solar(n)[0])(intervalo)
    graph(ax_graph1, velocidade_atualizado, reynolds_atualizado, 'blue', 'Velocidade (m/s)', 'Reynolds', velocidade, reynolds)

    nu_atualizado = np.vectorize(lambda n: solar(n)[9])(intervalo)
    temp_interna_atualizado = np.vectorize(lambda n: solar(n)[1])(intervalo)
    graph(ax_graph2, temp_interna_atualizado, nu_atualizado, 'red', 'Temperatura da Superfície Interna (°C)', 'Numero de Nusselt', temp_int, nu)

    area_atualizado = np.vectorize(lambda n: solar(n)[6])(intervalo)
    calor_incidente_atualizado = np.vectorize(lambda n: solar(n)[5])(intervalo)
    graph(ax_graph3, area_atualizado, calor_incidente_atualizado, 'purple', 'Área do Heliostato (m\u00b2)', 'Taxa de Calor Incidente (W)', area, calor_incidente)

    plt.draw()

def atualiza_slider(val):
    prop['d_tubo'] = slider_dTubo.val
    prop['d_topo'] = slider_dTopo.val
    prop['d_base'] = slider_dBase.val
    prop['altura'] = slider_altura.val
    prop['refletividade'] = slider_refletividade.val
    prop['espessura'] = slider_espessura.val
    atualiza_grafico()

def atualiza_material(label):
    global condutividade_material, emissividade_material, absortividade_material
    k_materiais = {'Cobre': 396.5, 'AISI 304': 14.86, 'Ferro': 80.52}
    e_materiais = {'Cobre': 0.061, 'AISI 304': 0.1963, 'Ferro': 0.34}
    a_materiais = {'Cobre': 0.80, 'AISI 304': 0.85, 'Ferro': 0.90}
    condutividade_material = k_materiais[label]
    emissividade_material = e_materiais[label]
    absortividade_material = a_materiais[label]
    atualiza_grafico()

#--------------------------------------------------------------------------------------------------------
prop = {'d_base':0.3,'d_topo':0.1,'d_tubo':0.05,'altura':0.3, 'refletividade': 0.9,'comprimento': 1, 'espessura':0.005, 'voltas':1}

condutividade_material = 396.5
emissividade_material = 0.061
absortividade_material = 0.80

# Cria o GridSpec
fig = plt.figure(num='Receptor Solar Helicoidal', figsize=(10,8))

gs = gridspec.GridSpec(4, 2, height_ratios=[3,2,1,1], width_ratios=[1,1]) # Definir Grids

#Adiciona o Gráfico Tridimensional
ax_helice = fig.add_subplot(gs[0,0], projection='3d')

#Adiciona o Gráfico Bidimensional
ax_solar = fig.add_subplot(gs[1,:])

#Adiciona seção do tubol
ax_tubo = fig.add_subplot(gs[0,1])


#Sliders ---------------------------------------------------------------------------------------------------
ax_slider_dTubo = plt.axes([0.25, 0.26, 0.2, 0.03])
slider_dTubo = Slider(ax_slider_dTubo, 'Diâmetro do Tubo (m)', 0.01, 0.1, valinit=0.05, valstep=0.001, color='orange')
slider_dTubo.on_changed(atualiza_slider)

ax_slider_espessura = plt.axes([0.6, 0.26, 0.15, 0.03])
slider_espessura = Slider(ax_slider_espessura, 'Espessura (m)', 0.001, 0.01, valinit=0.005, valstep=0.001, color='orange')
slider_espessura.on_changed(atualiza_slider)

ax_slider_dTopo = plt.axes([0.25, 0.22, 0.5, 0.03])
slider_dTopo = Slider(ax_slider_dTopo, 'Diâmetro do Topo (m)', 0.01, 0.5, valinit=0.1, valstep=0.01, color='green')
slider_dTopo.on_changed(atualiza_slider)

ax_slider_dBase = plt.axes([0.25, 0.18, 0.5, 0.03])
slider_dBase = Slider(ax_slider_dBase, 'Diâmetro da Base (m)', 0.01, 0.5, valinit=0.3, valstep=0.01, color='green')
slider_dBase.on_changed(atualiza_slider)

ax_slider_altura = plt.axes([0.25, 0.14, 0.5, 0.03])
slider_altura = Slider(ax_slider_altura, 'Altura (m)', 0.1, 0.5, valinit=0.3, valstep=0.01, color='green')
slider_altura.on_changed(atualiza_slider)

ax_slider_refletividade = plt.axes([0.25, 0.1, 0.5, 0.03])
slider_refletividade = Slider(ax_slider_refletividade, 'Refletividade do Heliostato', 0.01, 1, valinit=0.9, valstep=0.01, color='purple')
slider_refletividade.on_changed(atualiza_slider)

# TextBox ---------------------------------------------------------------------------------------------------
ax_calor = plt.axes([0.25, 0.06, 0.10, 0.03])
text_calor = TextBox(ax_calor, 'Taxa de Calor Útil (W)', initial=4000, textalignment='center', label_pad=0.05)

ax_radiacao = plt.axes([0.55, 0.06, 0.2, 0.03])
text_radiacao = TextBox(ax_radiacao, 'Radiação Incidente (W/m\u00b2)', initial=500, textalignment='center', label_pad=0.04)

ax_tempE = plt.axes([0.25, 0.01, 0.10, 0.03])
text_tempE = TextBox(ax_tempE, 'Temp. de Entrada do Ar (°C)', initial=20, textalignment='center', label_pad=0.06)

ax_tempS = plt.axes([0.55, 0.01, 0.2, 0.03])
text_tempS = TextBox(ax_tempS, 'Temp. de Saída do Ar (°C)', initial=350, textalignment='center', label_pad=0.06)

#Radios ---------------------------------------------------------------------------------------------------
ax_radio = plt.axes([0.025, 0.6, 0.15, 0.15])
radio = RadioButtons(ax_radio, ('Cobre', 'AISI 304', 'Ferro'))
radio.on_clicked(atualiza_material)

L_text = ax_helice.text2D(0.05, 1, f"Comprimento = {prop['comprimento']:.2f} m", transform=ax_helice.transAxes)
voltas_text = ax_helice.text2D(0.9, 1, f"Voltas = {prop['voltas']}", transform=ax_helice.transAxes)
solar_text = ax_helice.text2D(0.05, 0.91, f"Taxa de Calor Incidente = {(solar(prop['d_tubo'])[0])} W", transform=ax_helice.transAxes)
area_text = ax_helice.text2D(0.05, 0.82, f"Área Total dos Heliostatos = {(solar(prop['d_tubo'])[0])} m\u00b2", transform=ax_helice.transAxes)

sns.set_theme()
fig_graph = plt.figure(num='Gráfico', figsize=(8,8))
gs_graph = gridspec.GridSpec(3, 3, height_ratios=[1,1,1], width_ratios=[1,4,1])

ax_graph1 = fig_graph.add_subplot(gs_graph[0,1])
ax_graph2 = fig_graph.add_subplot(gs_graph[1,1])
ax_graph3 = fig_graph.add_subplot(gs_graph[2,1])

fig_graph.tight_layout()

atualiza_grafico()

plt.show()