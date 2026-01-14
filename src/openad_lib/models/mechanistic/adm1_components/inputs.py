def input_variables_ADM1(influent_state,i):

    # Input values (influente) 
    S_su_in     = influent_state['S_su'][i]         #kg COD.m^-3
    S_aa_in     = influent_state['S_aa'][i]         #kg COD.m^-3
    S_fa_in     = influent_state['S_fa'][i]         #kg COD.m^-3
    S_va_in     = influent_state['S_va'][i]         #kg COD.m^-3
    S_bu_in     = influent_state['S_bu'][i]         #kg COD.m^-3
    S_pro_in    = influent_state['S_pro'][i]        #kg COD.m^-3
    S_ac_in     = influent_state['S_ac'][i]         #kg COD.m^-3
    S_h2_in     = influent_state['S_h2'][i]         #kg COD.m^-3
    S_ch4_in    = influent_state['S_ch4'][i]        #kg COD.m^-3
    S_IC_in     = influent_state['S_IC'][i]         #kmole C.m^-3
    S_IN_in     = influent_state['S_IN'][i]         #kmole N.m^-3
    S_I_in      = influent_state['S_I'][i]          #kg COD.m^-3
    X_xc_in     = influent_state['X_xc'][i]         #kg COD.m^-3
    X_ch_in     = influent_state['X_ch'][i]         #kg COD.m^-3
    X_pr_in     = influent_state['X_pr'][i]         #kg COD.m^-3
    X_li_in     = influent_state['X_li'][i]         #kg COD.m^-3
    X_su_in     = influent_state['X_su'][i]         #kg COD.m^-3
    X_aa_in     = influent_state['X_aa'][i]         #kg COD.m^-3
    X_fa_in     = influent_state['X_fa'][i]         #kg COD.m^-3
    X_c4_in     = influent_state['X_c4'][i]         #kg COD.m^-3
    X_pro_in    = influent_state['X_pro'][i]        #kg COD.m^-3
    X_ac_in     = influent_state['X_ac'][i]         #kg COD.m^-3
    X_h2_in     = influent_state['X_h2'][i]         #kg COD.m^-3
    X_I_in      = influent_state['X_I'][i]          #kg COD.m^-3    
    S_cation_in = influent_state['S_cation'][i]     #kmole.m^-3
    S_anion_in  = influent_state['S_anion'][i]      #kmole.m^-3
    q_ad        = influent_state['q_ad'][i]         #kmole.m^-3

    ##Output vector with the necessary inputs for ADM1
    yin = [S_su_in,S_aa_in,S_fa_in,S_va_in,S_bu_in,S_pro_in,S_ac_in,S_h2_in,S_ch4_in,S_IC_in,S_IN_in,S_I_in,\
               X_xc_in,X_ch_in,X_pr_in,X_li_in,X_su_in,X_aa_in,X_fa_in,X_c4_in,X_pro_in,X_ac_in,X_h2_in,X_I_in,\
               S_cation_in,S_anion_in,q_ad]

    return yin
