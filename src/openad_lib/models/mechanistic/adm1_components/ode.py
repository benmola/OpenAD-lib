import numpy as np

# Function for calulating the derivatives related to ADM1 system of equations from the Rosen et al (2006) BSM2 report
def ADM1_ode(t, state_zero, yin,  V_liq, V_gas, Param):

    # Input state variables
    S_su_in,S_aa_in,S_fa_in,S_va_in,S_bu_in,S_pro_in,S_ac_in,S_h2_in,S_ch4_in,S_IC_in,S_IN_in,S_I_in,\
               X_xc_in,X_ch_in,X_pr_in,X_li_in,X_su_in,X_aa_in,X_fa_in,X_c4_in,X_pro_in,X_ac_in,X_h2_in,X_I_in,\
               S_cation_in,S_anion_in,q_ad = yin[0], yin[1], yin[2], yin[3], yin[4], yin[5], yin[6], yin[7], yin[8], yin[9],\
               yin[10], yin[11], yin[12], yin[13], yin[14], yin[15], yin[16], yin[17], yin[18], yin[19], yin[20], yin[21],\
               yin[22], yin[23], yin[24], yin[25],yin[26]

    # Constant physicochemical parameters
    R, T_base, T_op, k_p, p_atm, p_gas_h2o, k_hyd = Param

    # q_out       =  q_ad

    # Stoichiometric parameters (Rosen y Jeppsson, 2006)
    f_sI_xc     =  0.1
    f_xI_xc     =  0.2
    f_ch_xc     =  0.2
    f_pr_xc     =  0.2
    f_li_xc     =  0.3
    N_xc        =  0.0027
    N_I         =  0.0014       #kmole N.kg^-1COD
    N_aa        =  0.0114       #kmole N.kg^-1COD
    C_xc        =  0.02786      #kmole C.kg^-1COD
    C_sI        =  0.03         #kmole C.kg^-1COD
    C_ch        =  0.0313       #kmole C.kg^-1COD
    C_pr        =  0.03         #kmole C.kg^-1COD
    C_li        =  0.022        #kmole C.kg^-1COD
    C_xI        =  0.03         #kmole C.kg^-1COD
    C_su        =  0.0313       #kmole C.kg^-1COD
    C_aa        =  0.03         #kmole C.kg^-1COD
    f_fa_li     =  0.95
    C_fa        =  0.0217       #kmole C.kg^-1COD
    f_h2_su     =  0.19
    f_bu_su     =  0.13
    f_pro_su    =  0.27
    f_ac_su     =  0.41
    N_bac       =  0.0245       #kmole N.kg^-1COD
    C_bu        =  0.025        #kmole C.kg^-1COD
    C_pro       =  0.0268       #kmole C.kg^-1COD
    C_ac        =  0.0313       #kmole C.kg^-1COD
    C_bac       =  0.0313       #kmole C.kg^-1COD
    Y_su        =  0.1
    f_h2_aa     =  0.06
    f_va_aa     =  0.23
    f_bu_aa     =  0.26
    f_pro_aa    =  0.05
    f_ac_aa     =  0.40
    C_va        =  0.024        #kmole C.kg^-1COD
    Y_aa        =  0.08
    Y_fa        =  0.06
    Y_c4        =  0.06
    Y_pro       =  0.04
    C_ch4       =  0.0156       #kmole C.kg^-1COD
    Y_ac        =  0.05
    Y_h2        =  0.06

    # Biochemical parameters (Rosen y Jeppsson, 2006)
    k_dis       =  0.5          #d^-1
    k_hyd_ch    =  k_hyd        #d^-1
    k_hyd_pr    =  k_hyd        #d^-1
    k_hyd_li    =  k_hyd        #d^-1
    K_S_IN      =  10 ** -4     #M
    k_m_su      =  30           #d^-1
    K_S_su      =  0.5          #kgCOD.m^-3
    pH_UL_aa    =  5.5
    pH_LL_aa    =  4
    k_m_aa      =  50           #d^-1
    K_S_aa      =  0.3          #kgCOD.m^-3
    k_m_fa      =  6            #d^-1
    K_S_fa      =  0.4          #kgCOD.m^-3
    K_I_h2_fa   =  5 * 10 ** -6     #kgCOD.m^-3
    k_m_c4      =  20               #d^-1
    K_S_c4      =  0.2          #kgCOD.m^-3
    K_I_h2_c4   =  10 ** -5     #kgCOD.m^-3
    k_m_pro     =  13           #d^-1
    K_S_pro     =  0.1          #kgCOD.m^-3
    K_I_h2_pro  =  3.5 * 10 ** -6   #kgCOD.m^-3
    k_m_ac      =  8            #kgCOD.m^-3
    K_S_ac      =  0.15         #kgCOD.m^-3
    K_I_nh3     =  0.0018       #M
    pH_UL_ac    =  7
    pH_LL_ac    =  6
    k_m_h2      =  35           #d^-1
    K_S_h2      =  7 * 10 ** -6     #kgCOD.m^-3
    pH_UL_h2    =  6
    pH_LL_h2    =  5
    k_dec_X_su  =  0.02         #d^-1
    k_dec_X_aa  =  0.02         #d^-1
    k_dec_X_fa  =  0.02         #d^-1
    k_dec_X_c4  =  0.02         #d^-1
    k_dec_X_pro =  0.02         #d^-1
    k_dec_X_ac  =  0.02         #d^-1
    k_dec_X_h2  =  0.02  

    # Physico-chemical parameter values from the Rosen et al (2006) BSM2 report
    K_w         =  (10 ** -14.0) * np.exp((55900/(100*R))*(1/T_base - 1/T_op))      #M #2.08 * 10 ^ -14
    K_a_va      =  10 ** -4.86                                                      #M  ADM1 value = 1.38 * 10 ^ -5
    K_a_bu      =  10 ** -4.82                                                      #M #1.5 * 10 ^ -5
    K_a_pro     =  10 ** -4.88                                                      #M #1.32 * 10 ^ -5
    K_a_ac      =  10 ** -4.76                                                      #M #1.74 * 10 ^ -5
    K_a_co2     =  (10 ** -6.35) * np.exp((7646/(100 * R))*(1/T_base - 1/T_op))     #M #4.94 * 10 ^ -7
    K_a_IN      =  (10 ** -9.25) * np.exp((51965/(100 * R))*(1/T_base - 1/T_op))    #M #1.11 * 10 ^ -9
    k_A_B_va    =  10 ** 10                                                         #M^-1 * d^-1
    k_A_B_bu    =  10 ** 10                                                         #M^-1 * d^-1
    k_A_B_pro   =  10 ** 10                                                         #M^-1 * d^-1
    k_A_B_ac    =  10 ** 10                                                         #M^-1 * d^-1
    k_A_B_co2   =  10 ** 10                                                         #M^-1 * d^-1
    k_A_B_IN    =  10 ** 10                                                         #M^-1 * d^-1                                                            #m^3.d^-1.bar^-1 
    k_L_a       =  200.0                                                            #d^-1
    K_H_co2     =  0.035 * np.exp((-19410 / (100 * R))* (1 / T_base - 1 / T_op))    #Mliq.bar^-1 #0.0271
    K_H_ch4     =  0.0014 * np.exp((-14240/(100 * R)) * (1/T_base - 1/T_op))        #Mliq.bar^-1 #0.00116
    K_H_h2      =  (7.8 * 10 ** -4) * np.exp(-4180/(100 * R) * (1/T_base - 1/T_op)) #Mliq.bar^-1 #7.38*10^-4

   

    ##ADM1 model 
    ##differential equations from Rosen et al (2006) BSM2 report
   

    #Assignment of initial system values
    S_su = state_zero[0];       S_aa = state_zero[1];       S_fa = state_zero[2];           S_va = state_zero[3]
    S_bu = state_zero[4];       S_pro= state_zero[5];       S_ac = state_zero[6];           S_h2 = state_zero[7]
    S_ch4= state_zero[8];       S_IC = state_zero[9];       S_IN = state_zero[10];          S_I  = state_zero[11]
    X_xc = state_zero[12];      X_ch = state_zero[13];      X_pr = state_zero[14];          X_li = state_zero[15]
    X_su = state_zero[16];      X_aa = state_zero[17];      X_fa = state_zero[18];          X_c4= state_zero[19]
    X_pro= state_zero[20];      X_ac = state_zero[21];      X_h2 = state_zero[22];          X_I  = state_zero[23]
    S_cation = state_zero[24];  S_anion  = state_zero[25];  S_va_ion = state_zero[26];      S_bu_ion = state_zero[27]
    S_pro_ion= state_zero[28];  S_ac_ion = state_zero[29];  S_hco3_ion =  state_zero[30];   S_nh3   = state_zero[31]
    S_gas_h2 = state_zero[32];  S_gas_ch4= state_zero[33];  S_gas_co2= state_zero[34]

    #Partial pressures of gases
    p_gas_h2  = S_gas_h2 * R * T_op / 16
    p_gas_ch4 = S_gas_ch4* R * T_op / 64
    p_gas_co2 = S_gas_co2 * R * T_op
    
    #pH calculation
    te = S_cation + S_IN - S_nh3 - S_hco3_ion - S_ac_ion/64 - S_pro_ion/112 - S_bu_ion/160 - S_va_ion/208 - S_anion
    SH = -te/2 + np.sqrt(te**2 + 4*K_w)/2
    pH = -np.log10(SH + 1e-20)
    
    #inhibition values
    if pH < pH_UL_aa:
        I_pH_aa = np.exp(-3 * ((pH - pH_UL_aa)/(pH_UL_aa - pH_LL_aa))**2)
    else:
        I_pH_aa = 1

    if pH < pH_UL_ac:
        I_pH_ac  = np.exp(-3 * ((pH - pH_UL_ac)/(pH_UL_ac - pH_LL_ac))**2)
    else:
        I_pH_ac = 1

    if pH < pH_UL_h2:
        I_pH_h2 = np.exp(-3 * ((pH - pH_UL_h2)/(pH_UL_h2 - pH_LL_h2))**2)
    else:
        I_pH_h2 = 1
    
    I_IN_lim =  (1 / (1 + (K_S_IN / S_IN)))
    I_h2_fa =  (1 / (1 + (S_h2 / K_I_h2_fa)))
    I_h2_c4 =  (1 / (1 + (S_h2 / K_I_h2_c4)))
    I_h2_pro =  (1 / (1 + (S_h2 / K_I_h2_pro)))
    I_nh3 =  (1 / (1 + (S_nh3 / K_I_nh3)))
    
    I_5 =  (I_pH_aa * I_IN_lim)
    I_6 =  I_5 
    I_7 =  (I_pH_aa* I_IN_lim * I_h2_fa)
    I_8 =  (I_pH_aa * I_IN_lim * I_h2_c4)
    I_9 =  I_8 
    I_10 =  (I_pH_aa* I_IN_lim * I_h2_pro)
    I_11 =  (I_pH_ac* I_IN_lim * I_nh3)
    I_12 =  (I_pH_h2* I_IN_lim)

    # biochemical process rates from Rosen et al (2006) BSM2 report
    Rho_1 =  (k_dis * X_xc )     # Disintegration
    Rho_2 =  (k_hyd_ch * X_ch )  # Hydrolysis of carbohydrates
    Rho_3 =  (k_hyd_pr * X_pr )  # Hydrolysis of proteins
    Rho_4 =  (k_hyd_li * X_li )  # Hydrolysis of lipids
    Rho_5 =  (k_m_su * (S_su  / (K_S_su + S_su )) * X_su  * I_5 )    # Uptake of sugars
    Rho_6 =  (k_m_aa * (S_aa  / (K_S_aa + S_aa )) * X_aa  * I_6 )    # Uptake of amino-acids
    Rho_7 =  (k_m_fa * (S_fa  / (K_S_fa + S_fa )) * X_fa  * I_7 )    # Uptake of LCFA (long-chain fatty acids)
    Rho_8 =  (k_m_c4 * (S_va  / (K_S_c4 + S_va )) * X_c4  * (S_va  / (S_bu  + S_va  + 1e-6)) * I_8 )  # Uptake of valerate
    Rho_9  =  (k_m_c4 * (S_bu  / (K_S_c4 + S_bu )) * X_c4  * (S_bu  / (S_bu  + S_va  + 1e-6)) * I_9 )  # Uptake of butyrate
    Rho_10  =  (k_m_pro * (S_pro  / (K_S_pro + S_pro )) * X_pro  * I_10 )  # Uptake of propionate
    Rho_11  =  (k_m_ac * (S_ac  / (K_S_ac + S_ac )) * X_ac  * I_11 )  # Uptake of acetate
    Rho_12  =  (k_m_h2 * (S_h2  / (K_S_h2 + S_h2 )) * X_h2  * I_12 )  # Uptake of hydrogen
    Rho_13  =  (k_dec_X_su * X_su )     # Decay of X_su
    Rho_14  =  (k_dec_X_aa * X_aa )     # Decay of X_aa
    Rho_15  =  (k_dec_X_fa * X_fa )     # Decay of X_fa
    Rho_16  =  (k_dec_X_c4 * X_c4 )     # Decay of X_c4
    Rho_17  =  (k_dec_X_pro * X_pro )   # Decay of X_pro
    Rho_18  =  (k_dec_X_ac * X_ac )     # Decay of X_ac
    Rho_19  =  (k_dec_X_h2 * X_h2 )     # Decay of X_h2

    # acid-base rates for the BSM2 ODE implementation from Rosen et al (2006) BSM2 report
    Rho_A_4  =  (k_A_B_va * (S_va_ion  * (K_a_va + SH ) - K_a_va * S_va ))
    Rho_A_5  =  (k_A_B_bu * (S_bu_ion  * (K_a_bu + SH ) - K_a_bu * S_bu ))
    Rho_A_6  =  (k_A_B_pro * (S_pro_ion  * (K_a_pro + SH ) - K_a_pro * S_pro ))
    Rho_A_7  =  (k_A_B_ac * (S_ac_ion  * (K_a_ac + SH ) - K_a_ac * S_ac ))
    Rho_A_10  = (k_A_B_co2 * (S_hco3_ion  * (K_a_co2 + SH ) - K_a_co2 * S_IC ))
    Rho_A_11  = (k_A_B_IN * (S_nh3  * (K_a_IN + SH ) - K_a_IN * S_IN ))

    # gas transfer rates from Rosen et al (2006) BSM2 report
    Rho_T_8  =  max(0, k_L_a * (S_h2  - 16 * K_H_h2 * p_gas_h2 ))          
    Rho_T_9  =  max(0, k_L_a * (S_ch4  - 64 * K_H_ch4 * p_gas_ch4 ))            
    Rho_T_10  = max(0, k_L_a * ((S_IC -S_hco3_ion ) - K_H_co2 * p_gas_co2 ))

    #constants for eq #10
    s_1         = (-1 * C_xc + f_sI_xc * C_sI + f_ch_xc * C_ch + f_pr_xc * C_pr + f_li_xc * C_li + f_xI_xc * C_xI) 
    s_2         = (-1 * C_ch + C_su)
    s_3         = (-1 * C_pr + C_aa)
    s_4         = (-1 * C_li + (1 - f_fa_li) * C_su + f_fa_li * C_fa)
    s_5         = (-1 * C_su + (1 - Y_su) * (f_bu_su * C_bu + f_pro_su * C_pro + f_ac_su * C_ac) + Y_su * C_bac)
    s_6         = (-1 * C_aa + (1 - Y_aa) * (f_va_aa * C_va + f_bu_aa * C_bu + f_pro_aa * C_pro + f_ac_aa * C_ac) + Y_aa * C_bac)
    s_7         = (-1 * C_fa + (1 - Y_fa) * 0.7 * C_ac + Y_fa * C_bac)
    s_8         = (-1 * C_va + (1 - Y_c4) * 0.54 * C_pro + (1 - Y_c4) * 0.31 * C_ac + Y_c4 * C_bac)
    s_9         = (-1 * C_bu + (1 - Y_c4) * 0.8 * C_ac + Y_c4 * C_bac)
    s_10        = (-1 * C_pro + (1 - Y_pro) * 0.57 * C_ac + Y_pro * C_bac)
    s_11        = (-1 * C_ac + (1 - Y_ac) * C_ch4 + Y_ac * C_bac)
    s_12        = ((1 - Y_h2) * C_ch4 + Y_h2 * C_bac)
    s_13        = (-1 * C_bac + C_xc) 

 
    # differential equations 1 to 12 (soluble matter)
    diff_S_su  = 1/V_liq *(q_ad*S_su_in + np.dot(q_ad,S_su)) + Rho_2  + (1 - f_fa_li) * Rho_4  - Rho_5   # eq1                     
    diff_S_aa  = 1/V_liq *(q_ad*S_aa_in + np.dot(q_ad,S_aa)) + Rho_3  - Rho_6   # eq2
    diff_S_fa  = 1/V_liq *(q_ad*S_fa_in + np.dot(q_ad,S_fa)) + (f_fa_li * Rho_4 ) - Rho_7   # eq3
    diff_S_va  = 1/V_liq *(q_ad*S_va_in + np.dot(q_ad,S_va)) + (1 - Y_aa) * f_va_aa * Rho_6  - Rho_8   # eq4
    diff_S_bu  = 1/V_liq *(q_ad*S_bu_in + np.dot(q_ad,S_bu)) + (1 - Y_su) * f_bu_su * Rho_5  +\
                   (1 - Y_aa) * f_bu_aa * Rho_6  - Rho_9   # eq5
    diff_S_pro  = 1/V_liq *(q_ad*S_pro_in + np.dot(q_ad,S_pro)) + (1 - Y_su) * f_pro_su * Rho_5  +\
                    (1 - Y_aa) * f_pro_aa * Rho_6  + (1 - Y_c4) * 0.54 * Rho_8  - Rho_10   # eq6
    diff_S_ac  = 1/V_liq *(q_ad*S_ac_in + np.dot(q_ad,S_ac)) + (1 - Y_su) * f_ac_su * Rho_5  +\
                   (1 - Y_aa) * f_ac_aa * Rho_6  + (1 - Y_fa) * 0.7 * Rho_7  + (1 - Y_c4) * 0.31 * Rho_8  + (1 - Y_c4) * 0.8 * Rho_9  + (1 - Y_pro) * 0.57 * Rho_10  - Rho_11   # eq7
    diff_S_h2  = 1/V_liq *(q_ad*S_h2_in + np.dot(q_ad,S_h2)) + (1 - Y_su) * f_h2_su * Rho_5  +\
                   (1 - Y_aa) * f_h2_aa * Rho_6  + (1 - Y_fa) * 0.3 * Rho_7  + (1 - Y_c4) * 0.15 * Rho_8  + (1 - Y_c4) * 0.2 * Rho_9  + (1 - Y_pro) * 0.43 * Rho_10  - Rho_12  - Rho_T_8  # eq8
    diff_S_ch4  = 1/V_liq  * (q_ad*S_ch4_in + np.dot(q_ad,S_ch4)) + (1 - Y_ac) * Rho_11  +\
                    (1 - Y_h2) * Rho_12  - Rho_T_9   # eq9
    ## eq10 start##
    Sigma   =  (s_1 * Rho_1  + s_2 * Rho_2  + s_3 * Rho_3  + s_4 * Rho_4  + s_5 * Rho_5  + s_6 * Rho_6  +\
                 s_7 * Rho_7  + s_8 * Rho_8  + s_9 * Rho_9  + s_10 * Rho_10  + s_11 * Rho_11  +\
                 s_12 * Rho_12  + s_13 * (Rho_13  + Rho_14  + Rho_15  + Rho_16  + Rho_17  + Rho_18  + Rho_19 ))
    diff_S_IC  = 1/V_liq  * (q_ad*S_IC_in + np.dot(q_ad,S_IC)) - Sigma - Rho_T_10 
    ## eq10 end##
    diff_S_IN  = 1/V_liq  * (q_ad*S_IN_in + np.dot(q_ad,S_IN)) + (N_xc - f_xI_xc * N_I - f_sI_xc * N_I -f_pr_xc * N_aa) * Rho_1  -\
                   Y_su * N_bac * Rho_5  + (N_aa - Y_aa * N_bac) * Rho_6  - Y_fa * N_bac * Rho_7  - Y_c4 * N_bac * Rho_8  -\
                   Y_c4 * N_bac * Rho_9  - Y_pro * N_bac * Rho_10  - Y_ac * N_bac * Rho_11  - Y_h2 * N_bac * Rho_12  +\
                   (N_bac - N_xc) * (Rho_13  + Rho_14  + Rho_15  + Rho_16  + Rho_17  + Rho_18  + Rho_19 ) # eq11 
    diff_S_I  = 1/V_liq  * (q_ad*S_I_in + np.dot(q_ad,S_I)) + f_sI_xc * Rho_1   # eq12

    # Differential equations 13 to 24 (particulate matter)
    diff_X_xc  = 1/ V_liq  * (q_ad*X_xc_in + np.dot(q_ad,X_xc)) - Rho_1  + Rho_13  + Rho_14  + Rho_15  +\
                   Rho_16  + Rho_17  + Rho_18  + Rho_19   # eq13 
    diff_X_ch  = 1/ V_liq  * (q_ad*X_ch_in + np.dot(q_ad,X_ch)) + f_ch_xc * Rho_1  - Rho_2   # eq14 
    diff_X_pr  = 1/ V_liq  * (q_ad*X_pr_in + np.dot(q_ad,X_pr)) + f_pr_xc * Rho_1  - Rho_3   # eq15 
    diff_X_li  = 1/ V_liq  * (q_ad*X_li_in + np.dot(q_ad,X_li)) + f_li_xc * Rho_1  - Rho_4   # eq16 
    diff_X_su  = 1/ V_liq  * (q_ad*X_su_in + np.dot(q_ad,X_su)) + Y_su * Rho_5  - Rho_13   # eq17
    diff_X_aa  = 1/ V_liq  * (q_ad*X_aa_in + np.dot(q_ad,X_aa)) + Y_aa * Rho_6  - Rho_14   # eq18
    diff_X_fa  = 1/ V_liq  * (q_ad*X_fa_in + np.dot(q_ad,X_fa)) + Y_fa * Rho_7  - Rho_15   # eq19
    diff_X_c4  = 1/ V_liq  * (q_ad*X_c4_in + np.dot(q_ad,X_c4)) + Y_c4 * Rho_8  + Y_c4 * Rho_9  - Rho_16   # eq20
    diff_X_pro  = 1/ V_liq  * (q_ad*X_pro_in + np.dot(q_ad,X_pro)) + Y_pro * Rho_10  - Rho_17   # eq21
    diff_X_ac  = 1/ V_liq  * (q_ad*X_ac_in + np.dot(q_ad,X_ac)) + Y_ac * Rho_11  - Rho_18   # eq22
    diff_X_h2  = 1/ V_liq  * (q_ad*X_h2_in + np.dot(q_ad,X_h2)) + Y_h2 * Rho_12  - Rho_19   # eq23
    diff_X_I  = 1/ V_liq  * (q_ad*X_I_in + np.dot(q_ad,X_I)) + f_xI_xc * Rho_1   # eq24 

    # Differential equations 25 and 26 (cations and anions)
    diff_S_cation  = 1/ V_liq  * (q_ad*S_cation_in + np.dot(q_ad,S_cation))  # eq25
    diff_S_anion  = 1/ V_liq  * (q_ad*S_anion_in + np.dot(q_ad,S_anion))  # eq26

    # Differential equations 27 to 32 (ion states)
    diff_S_va_ion  = -Rho_A_4   # eq27
    diff_S_bu_ion  = -Rho_A_5   # eq28
    diff_S_pro_ion  = -Rho_A_6  # eq29
    diff_S_ac_ion  = -Rho_A_7   # eq30
    diff_S_hco3_ion  = -Rho_A_10   # eq31
    diff_S_nh3  = -Rho_A_11   # eq32

    # Gas phase equations: Differential equations 33 to 35
    p_gas  = p_gas_h2  + p_gas_ch4  + p_gas_co2  + p_gas_h2o
    q_gas  = max(0, (k_p * (p_gas  - p_atm))*p_gas /p_atm)
    
    diff_S_gas_h2  = (q_gas  * -1 * S_gas_h2 ) / V_gas  + (Rho_T_8  * V_liq / V_gas )  # eq33
    diff_S_gas_ch4  = (q_gas  * -1 * S_gas_ch4 ) / V_gas  + (Rho_T_9  * V_liq / V_gas )  # eq34
    diff_S_gas_co2  = (q_gas  * -1 * S_gas_co2 ) / V_gas  + (Rho_T_10  * V_liq / V_gas )  # eq35
                 

    ## Generation of derivative vector
    diff = np.array([])

    
    x = [diff_S_su , diff_S_aa , diff_S_fa , diff_S_va , diff_S_bu , diff_S_pro , diff_S_ac , diff_S_h2 , diff_S_ch4 ,\
         diff_S_IC , diff_S_IN , diff_S_I , diff_X_xc , diff_X_ch , diff_X_pr , diff_X_li , diff_X_su , diff_X_aa ,\
         diff_X_fa , diff_X_c4 , diff_X_pro , diff_X_ac , diff_X_h2 , diff_X_I , diff_S_cation , diff_S_anion ,\
         diff_S_va_ion , diff_S_bu_ion , diff_S_pro_ion , diff_S_ac_ion , diff_S_hco3_ion , diff_S_nh3 , diff_S_gas_h2 ,\
         diff_S_gas_ch4 , diff_S_gas_co2 ]

    diff = np.append(diff, x)

    return diff
