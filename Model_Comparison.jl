"""
        
        Neural TECMD model implimentation in Julia.
        Mechanistic model : First order ECM model with diffusion ( Xiong-2022) and thermal (lumped model) blocks added.
        Data driven model : 2 FFNN for voltage and temperature correction.

        Implimented by Jishnu Ayyangatu Kuzhiyil
        PhD @ WMG, University of Warwick


       [1] R. Xiong, J. Huang, Y. Duan, W. Shen, Enhanced Lithium-ion battery model considering 
           critical surface charge behavior, Applied Energy 314 (2022) 118915.doi:10.1016/J.APENERGY.2022.118915

"""

"""______________________________________________________Activating the virtual environment _______________________________________________________"""

using Pkg
Pkg.activate(".")
Pkg.instantiate()


"""_____________________________________________________Importing required packages and data____________________________________________________________________"""

    using Plots, JLD2, Statistics, ComponentArrays, Interpolations,Plots.PlotMeasures
    using DifferentialEquations, Lux, DiffEqFlux
    pyplot()
    

    #Load the data
    C_rate = "1C"   # Options are "0p10C", "0p5C", "1C", "2C"
    Temp =25      #Options are 0,10,25,45
    Do_plot = true  # To plot the experimental data
    SOC_ini =100.0  

    #This function find the index of the point in the dataset where voltage hold starts at the end of charging. 
    function find_voltage_hold_start_index(voltage_data; voltage_tolerance=1e-2, threshold=1e-5)
        for i in 2:length(voltage_data)-1
        
            if abs(voltage_data[i] - 4.2) < voltage_tolerance
                # Check if the voltage is constant around this point
                if abs(voltage_data[i] - voltage_data[i+1]) < threshold && abs(voltage_data[i] - voltage_data[i-1]) < threshold
                    return i
                end
            end

        end
        return -1 
    end


    data_file = load("Datasets.jld2")["Datasets"]  #Load the JLD2 file containing the data
    data = data_file["$(C_rate)_T$(Temp)"]         #Extract the data for the required C-rate and temperature
    n = find_voltage_hold_start_index(data["voltage"])   #The data contains voltage hold at the end, this function finds the index where the hold starts, so data before this index is used. 
   # n=length(data["voltage"])
    t_exp = data["time"][1:n]
    I_exp = data["current"][1:n]
    T_exp = data["temperature"][1:n]
    V_exp = data["voltage"][1:n]

    if Do_plot == true
       p1= plot(t_exp,V_exp, label = "Experimental data", xlabel = "Time (s)", ylabel = "Voltage (V)", title = "Voltage vs time")
       p2 = plot(t_exp,T_exp, label = "Experimental data", xlabel = "Time (s)", ylabel = "Temperature (K)", title = "Temperature vs time")
       p3 = plot(t_exp,I_exp, label = "Experimental data", xlabel = "Time (s)", ylabel = "Current (A)", title = "Current vs time")
       display(plot(p1,p2,p3,layout = (3,1), size = (800,800)))
    end


"""___________________________________________________________________Model parameters___________________________________________________________________________"""


    """Circuit and diffusion parameters"""   #These parameters are reported in the paper (Table 3)
    R₀_ref = 20.611 *1e-3  #Ω
    R_ref = 12.5041 *1e-3  #Ω
    τ_ref = 0.189          #s
    θ_ref = 41.6           #100s
    T_ref = 298.15         #K
    Q = 4.84*3600          #As

    Ea_R₀ = 8359.94 #J/mol
    Ea_R =  9525.26 #J/mol
    Ea_θ = 25000.0  #J/mol

    """Thermal parameters"""
    C₁ = -0.0015397895     
    C₂ = 0.020306583

    T_ini = T_exp[1] #K  Initial temperature of the cell
    T∞ = T_exp[1]    #K  Ambient temperature

    """Neural network parameters"""
    NN_para_volt = load("NN_para_volt.jld2")  #This file contains the parameters of the neural network for voltage correction (NN1)
    NN_para_temp = load("NN_para_temp.jld2")  #This file contains the parameters of the neural network for temperature correction (NN2)

    Para_volt = NN_para_volt["NN_Vals"]["Parameters"]
    st_volt= NN_para_volt["NN_Vals"]["State"]
    Para_temp = NN_para_temp["NN_Vals"]["Parameters"]
    st_temp= NN_para_temp["NN_Vals"]["State"]

    NN_volt =  Lux.Chain(Lux.Dense(4, 20, tanh), Lux.Dense(20, 5, tanh), Lux.Dense(5, 1))   #here NN1 is defined 
    NN_temp = Lux.Chain(Lux.Dense(4, 20, tanh), Lux.Dense(20, 15, tanh), Lux.Dense(15, 1))  #here NN2 is defined
    

    """OCV function approximation"""   #this is the polynomial approximation of OCV as function of SOC
    @inline function OCV_LGM50_full_cell(SOC) # SOC ∈ [0,100] #OCP function
            
        if SOC < -2.5 #below 2.5% SOC a flat extrapolation is used.
            SOC = -2.5
        end 
        OCV =@.(3.132620508717038 + 0.03254176983890392*SOC - 0.003901708378797115*SOC^2 + 
        0.001382745468407752*SOC^3 - 0.00026033289859565573*SOC^4 + 2.7051704798205416e-5*SOC^5 -
        1.753670407892406e-6*SOC^6 + 7.619006342039215e-8*SOC^7 - 2.3099431268369483e-9*SOC^8 +
        4.990071985886017e-11*SOC^9 - 7.728673298552951e-13*SOC^10 + 8.517228717429399e-15*SOC^11 - 
        6.51737840308998e-17*SOC^12 + 3.2902385347157566e-19*SOC^13 - 9.851142596586927e-22*SOC^14 +
        1.3245328408180436e-24*SOC^15) #Broadcasted for faster calculation

    end 

    I_function =  LinearInterpolation(t_exp , I_exp , extrapolation_bc=Linear()) 


"""_________________________________________________________Defining the model____________________________________________________________________________________"""



    function Model!(du,u,p,t,)  #This is the model function for the mechanistic model and the UDE model. The UDE model is obtained by setting p=1.0 and the mechanistic model is obtained by setting p=0.0

        
        #Terminology______________________________________________________________
        I = I_function(t)
        SOC,ΔSOC,q1,q2 = u[1],u[2],u[3],u[4]
        V_MM,V_NN = u[5],u[6]
        T = u[7]
        C₂_updated = (1 - 0.35 *p)*C₂  #p=0 for mechanistic and p=1 for UDE

        #Circuit parameters________________________________________________________
        R₀ = R₀_ref * exp((Ea_R₀/8.314)*(1/T - 1/T_ref))
        R = R_ref * exp((Ea_R/8.314)*(1/T - 1/T_ref))
        τ = τ_ref
        θ = θ_ref * exp((Ea_θ/8.314)*(1/T - 1/T_ref))

        #SOCdynamics______________________________________________________________
        dSOC = (-(100.0/Q)*I ) # Coulomb counting, SOC ∈ [0,100]
        dΔSOC = (-5.94/θ)     * ΔSOC  + q1 - (1100.0/Q)    *  I
        dq1   = (-4.5045/θ^2) * ΔSOC  + q2 - (1716.0/θ/Q)  *  I
        dq2   = (-0.6757/θ^3) * ΔSOC       - (450.5/θ^2/Q) *  I

        SOC_surf = SOC + ΔSOC

        #Voltage dynamics_________________________________________________________
        dV_MM = (-1.0/τ) * (V_MM + R*I)
        dV_NN = p * NN_volt([T,SOC_surf,I,V_NN],Para_volt,st_volt)[1][1] 

        V_over = V_MM + 1e-3 * V_NN - I*R₀

        #Temperature dynamics______________________________________________________
        dT = C₁ * (T - T∞) - C₂_updated * I * (p * NN_temp([ T ; SOC ; I ;0.0],Para_temp, st_temp)[1][1] + V_over)

        du .= [dSOC,dΔSOC,dq1,dq2,dV_MM,dV_NN,dT]

    end 

    #Initial conditions____________________________________________________________
  
    u0 = [SOC_ini,0.0,0.0,0.0,0.0,0.0,T_ini]  #Initial condition for the state vector 
    tspan = (0.0,t_exp[end])    #Duration of the simulation

    prob_UDE= ODEProblem(Model!,u0,tspan,1.0) #p=1.0 for UDE
    prob_MM = ODEProblem(Model!,u0,tspan,0.0) #p=0.0 for mechanistic model
    sol_UDE = solve(prob_UDE,Tsit5(), reltol=1e-6, abstol=1e-6, saveat = t_exp)
    sol_MM = solve(prob_MM,Tsit5(), reltol=1e-6, abstol=1e-6, saveat = t_exp)
    sol_array_UDE= Array(sol_UDE)
    sol_array_MM= Array(sol_MM)

    #Voltage Calculation___________________________________________________________
    V_UDE = OCV_LGM50_full_cell.(sol_array_UDE[1,:] .+ sol_array_UDE[2,:]) .+ sol_array_UDE[5,:] .+ 1e-3 .* sol_array_UDE[6,:] .- I_exp.* R₀_ref .* exp.((Ea_R₀/8.314).*(1 ./ sol_array_UDE[7,:] .- 1/T_ref))
    V_MM = OCV_LGM50_full_cell.(sol_array_MM[1,:] .+ sol_array_MM[2,:]) .+ sol_array_MM[5,:] .+ 1e-3 .* sol_array_MM[6,:] .- I_exp.* R₀_ref .* exp.((Ea_R₀/8.314).*(1 ./ sol_array_MM[7,:] .- 1/T_ref))



"""_________________________________________________________Analysing the results____________________________________________________________________________________"""


    V_RMSE_UDE = 1000.0 * sqrt(sum(abs2,V_exp .- V_UDE)/length(V_exp))
    V_RMSE_MM = 1000.0 * sqrt(sum(abs2,V_exp .- V_MM)/length(V_exp))
    T_RMSE_UDE = sqrt(sum(abs2,T_exp .- sol_array_UDE[7,:])/length(T_exp))
    T_RMSE_MM = sqrt(sum(abs2,T_exp .- sol_array_MM[7,:])/length(T_exp))
    V_UDE_max = maximum(abs.(V_UDE-V_exp))*1e3
    V_MM_max = maximum(abs.(V_MM-V_exp))*1e3
    T_UDE_max = maximum(abs.(T_exp .- sol_array_UDE[7,:]))
    T_MM_max = maximum(abs.(T_exp .- sol_array_MM[7,:]))

   println("The simulated condition correspond to $C_rate  discharge and C/3 charge at $Temp °C")
   
    println("Voltage RMSE : UDE = $V_RMSE_UDE mV, MM = $V_RMSE_MM mV and percentage improvement is $(100*(V_RMSE_MM - V_RMSE_UDE)/V_RMSE_MM) %")
    println("Temperature RMSE : UDE = $T_RMSE_UDE degC, MM = $T_RMSE_MM degC and percentage improvement is $(100*(T_RMSE_MM - T_RMSE_UDE)/T_RMSE_MM) %")

    println("Voltage max error : UDE = $V_UDE_max mV, MM = $V_MM_max mV and percentage improvement is $(100*(V_MM_max - V_UDE_max)/V_MM_max) %")
    println("Temperature max error : UDE = $T_UDE_max degC, MM = $T_MM_max degC and percentage improvement is $(100*(T_MM_max - T_UDE_max)/T_MM_max) %")



"""_________________________________________________________Plotting the Voltage comparison___________________________________________________________________________________""" 

# Define consistent attributes to maintain uniformity
font_size =20
label_font_size = 22
line_width = 2.8
legend_font_size = 12
margin_space = 8mm

# Dimensions for half the width of an A4 sheet
width_mm = 105 *0.8 # width in millimeters
height_mm = 105/2 *0.8# height=1/2 width in millimeters
dpi = 300
width_px = round(Int, width_mm * dpi / 25.4)  # converting mm to inches
height_px = round(Int, height_mm * dpi / 25.4)  # converting mm to inches
plot_size = (width_px, height_px)

    V_plot = plot(
        t_exp/3600.0, V_exp, 
        label="Experimental data", 
        lw=line_width, 
        linecolor=:black, 
        legend=false,
        minorgrid=true,
        legendfontsize=legend_font_size, 
        legendmargin=5mm, 
        framestyle=:box, 
        left_margin=[5mm 0mm],
        size=plot_size, 
        titlefontsize=font_size + 2, 
        xtickfontsize=font_size, 
        ytickfontsize=font_size,
        xlabelfontsize=label_font_size, 
        ylabelfontsize=label_font_size, 
        margin=margin_space,
    )

    plot!(
        V_plot, t_exp/3600.0, V_UDE, 
        label="Neural TECMD", 
        lw=line_width, 
        linestyle=:dashdot, 
        linecolor=:red
    )

    plot!(
        V_plot, t_exp/3600.0, V_MM, 
        label="TECMD", 
        lw=line_width, 
        linestyle=:dashdot, 
        linecolor=:blue,
        xlabel="Time (h)", 
        ylabel="Voltage (V)",
        dpi=300)



"""_________________________________________________________Plotting the Temperature comparison___________________________________________________________________________________"""

(Temp==45) ? Temp_lowlim=round(minimum(T_exp.-273.15)) : Temp_lowlim=Temp
# Define consistent attributes to maintain uniformity
# font_size = 13
# label_font_size = 15
# line_width = 2
legend_font_size = 20
# margin_space = 8mm

# Dimensions for half the width of an A4 sheet
width_mm = 105 * 0.8 # width in millimeters
height_mm = 105/2 * 0.8 # height = 1/2 width in millimeters
dpi = 300
width_px = round(Int, width_mm * dpi / 25.4)  # converting mm to inches
height_px = round(Int, height_mm * dpi / 25.4)  # converting mm to inches
plot_size = (width_px, height_px)

    # Plotting temperature data
    T_plot = plot(
        t_exp/3600.0,
        T_exp .- 273.15, 
        label="Experimental data", 
        lw=line_width, 
        linecolor=:black, 
        xlabel="Time (h)", 
        ylabel="Cell temperature (°C)",
        legend=:topright, 
        minorgrid=true,
        legendfontsize=legend_font_size, 
        framestyle=:box,
        size=plot_size, 
        #title="Temperature prediction",
        titlefontsize=font_size + 2, 
        xtickfontsize=font_size, 
        ytickfontsize=font_size,
        xlabelfontsize=label_font_size, 
        ylabelfontsize=label_font_size, 
        margin=margin_space
    )

    plot!(
        T_plot, 
        t_exp/3600.0, 
        sol_array_UDE[7,:] .- 273.15, 
        label="Neural TECMD", 
        lw=line_width, 
        linestyle=:dashdot, 
        linecolor=:red,
        legend=:topright
    )

    plot!(
        T_plot, 
        t_exp/3600.0, 
        sol_array_MM[7,:] .- 273.15, 
        label="TECMD", 
        lw=line_width, 
        linestyle=:dashdot, 
        linecolor=:blue,
        dpi=300,
        legend=:topright, 
    )




plot(V_plot,T_plot,layout = (2,1), size = (800,800))

