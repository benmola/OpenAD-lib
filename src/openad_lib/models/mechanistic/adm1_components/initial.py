## Initial conditions ADM1
## Material with 8% TS content
Ssu_0      = 0.001552067        # Soluble monosaccharides             # kgCOD m^-3
Saa_0      = 0.001402758        # Soluble amino acids                # kgCOD m^-3
Sfa_0      = 0.012982557        # Soluble LCFA                       # kgCOD m^-3
Sva_0      = 0.001935812        # Soluble valerate                   # kgCOD m^-3
Sbu_0      = 0.002544151        # Soluble butyrate                   # kgCOD m^-3
Spro_0     = 0.001865727        # Soluble propionate                 # kgCOD m^-3
Sac_0      = 6.003813344        # Soluble acetate                    # kgCOD m^-3
Sh2_0      = 3.92727E-08        # Soluble hydrogen                   # kgCOD m^-3
Sch4_0     = 0.38850914         # Soluble methane                    # kgCOD m^-3
SIC_0      = 0.113359571        # Soluble inorganic carbon           # kmole C m^-3
SIN_0      = 0.182316373        # Soluble inorganic nitrogen         # kmole N m^-3
SI_0       = 33.22385934        # Soluble inert materials            # kgCOD m^-3
Xc_0       = 0.266338267        # Particulate compounds              # kgCOD m^-3
Xch_0      = 0.002664987        # Particulate carbohydrates           # kgCOD m^-3
Xpr_0      = 0.002664987        # Particulate proteins               # kgCOD m^-3
Xli_0      = 0.003997475        # Particulate lipids                 # kgCOD m^-3
Xsu_0      = 0.02               # Biomass degrading sugars          # kgCOD m^-3
Xaa_0      = 0.02               # Biomass degrading amino acids     # kgCOD m^-3
Xfa_0      = 0.02               # Biomass degrading LCFA            # kgCOD m^-3
Xc4_0      = 0.012              # Biomass degrading valerate and butyrate # kgCOD m^-3
Xpro_0     = 0.012              # Biomass degrading propionate      # kgCOD m^-3
Xac_0      = 0.012              # Biomass degrading acetate         # kgCOD m^-3
Xh2_0      = 0.012              # Biomass degrading hydrogen       # kgCOD m^-3
XI_0       = 91.4605372         # Particulate inerts                # kgCOD m^-3
Scat_0     = 0.04               # Soluble cations                   # kmole m^-3
San_0      = 0.02               # Soluble anions                    # kmole m^-3
Sava_0     = 0.001932849        # Soluble valeric acid              # kgCOD m^-3
Sabu_0     = 0.002540594        # Soluble butyric acid              # kgCOD m^-3
Sapro_0    = 0.001862736        # Soluble propanoic acid            # kgCOD m^-3
Saac_0     = 5.6188             # Soluble acetic acid               # kgCOD m^-3
Shco3_0    = 0.107390911        # Soluble bicarbonate               # kmole C m^-3
Snh3_0     = 0.007089683        # Soluble ammonia                   # kmole N m^-3
Sgas_h2_0  = 0                  # Gaseous hydrogen                   # kgCOD m^-3
Sgas_ch4_0 = 0                  # Gaseous methane                    # kgCOD m^-3
Sgas_co2_0 = 0                  # Gaseous carbon dioxide             # kmole C m^-3

# CI // Initial conditions of state variables for each CSTR
CI=[Ssu_0, Saa_0, Sfa_0, Sva_0, Sbu_0, Spro_0, Sac_0, Sh2_0, Sch4_0, SIC_0, SIN_0, SI_0,
    Xc_0, Xch_0, Xpr_0, Xli_0, Xsu_0, Xaa_0, Xfa_0, Xc4_0, Xpro_0, Xac_0, Xh2_0, XI_0,
    Scat_0, San_0, Sava_0, Sabu_0, Sapro_0, Saac_0, Shco3_0, Snh3_0, Sgas_h2_0, Sgas_ch4_0, Sgas_co2_0]
