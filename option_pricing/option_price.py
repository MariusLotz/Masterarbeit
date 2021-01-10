import QuantLib as ql
from math import exp

def price_call(spot_price=100, strike_price=100, volatility=0.3, risk_free_rate=0, dividend_rate=0.1,
          spot_day=1, spot_month=11, spot_year=2020 , mat_day=1, mat_month=11, mat_year=2021,
               engine=0):
    """ Derives option prices with quantlib module:
        engine=0: CRR method American Option
        engine=1: MC method
        engine=2: finite difference method
        engine=3: Barone-Adesi-Whaley method
        engine=4: Black Formula for European Options
        engine=5: European Analytic close to engine
        engine=6: CRR method European Option"""

    ### Time&date setting
    day_count = ql.Actual365Fixed()
    calendar = ql.UnitedStates()
    calculation_date = ql.Date(spot_day, spot_month, spot_year)
    ql.Settings.instance().evaluationDate = calculation_date
    maturity_date = ql.Date(mat_day, mat_month, mat_year)

    ### Underlaying stochastic process
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
    flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, risk_free_rate, day_count))
    dividend_yield = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, dividend_rate, day_count))
    flat_vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(calculation_date, calendar, volatility, day_count))
    bsm_process = ql.BlackScholesMertonProcess(spot_handle,
                                               dividend_yield,
                                               flat_ts,
                                               flat_vol_ts)

    ### Options specifications:
    option_type = ql.Option.Call
    settlement = calculation_date
    payoff = ql.PlainVanillaPayoff(option_type, strike_price)
    # for European:
    eu_exercise = ql.EuropeanExercise(maturity_date)
    european_option = ql.VanillaOption(payoff, eu_exercise)
    # for American:
    am_exercise = ql.AmericanExercise(settlement, maturity_date)
    american_option = ql.VanillaOption(payoff, am_exercise)

    ### Pricing engine:
    if engine==0:
        # CRR method
        steps = 9999
        binomial_engine = ql.BinomialVanillaEngine(bsm_process, "crr", steps)
        american_option.setPricingEngine(binomial_engine)
    elif engine==1:
        # MC method
        steps=999
        number_paths=9999
        rng = "pseudorandom"  # could use "lowdiscrepancy"
        MC_engine = ql.MCAmericanEngine(bsm_process, rng, steps, requiredSamples=number_paths)
        american_option.setPricingEngine(MC_engine)
    elif engine==2:
        # finite difference method ...???
        tGrid, xGrid = 9999, 999
        FD_engine = ql.FdBlackScholesVanillaEngine(bsm_process, tGrid, xGrid)
        american_option.setPricingEngine(FD_engine)
    elif engine==3:
        # Barone-Adesi-Whaley method
        american_option.setPricingEngine(ql.BaroneAdesiWhaleyApproximationEngine(bsm_process))
    elif engine==4:
        # Black Formula for European Options
        # \warning instead of volatility it uses standard deviation,
        # i.e. volatility*sqrt(timeToMaturity)
        tau = ql.Actual365Fixed().yearFraction(calculation_date,maturity_date)
        #print(tau)
        forward_price = spot_price * exp((risk_free_rate-dividend_rate)*tau)
        #black_method = ql.BlackCalculator(strike_price, forward_price, volatility)
        black_formula = ql.blackFormula(option_type,strike_price,forward_price,volatility*(tau)**0.5,
                                        exp(-risk_free_rate*tau),)
        return black_formula
    elif engine==5:
        european_option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))
        return european_option.NPV()

    elif engine==6:
        steps = 9999
        european_option.setPricingEngine(ql.BinomialVanillaEngine(bsm_process,"crr",steps))
        return  european_option.NPV()
    else:
        print("Please choose engine between 0 and 6.")
    return american_option.NPV()

