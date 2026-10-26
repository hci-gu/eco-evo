"""Effect of each candidate correction on the benthivores' required intake.

    python3 options.py

Pure arithmetic on the year-1 probe values (tables/accounting_stdout.txt):
action mix eat 0.957 / move 0.041 / rest 0.002, fill 0.722, RM 3.6 MJ/t/tick
(Q10 annual mean 1.000), production P 0.708 /yr at the realised fill and
0.821 /yr in the budget. For each variant it prints the assimilated energy
needed per t fish per yr (costs + production), the infauna Q/B that pays it
on a 99 % infauna diet, and the infauna M2 that follows at benthivores 1.53
t/km2 and infauna B* 93.6 t/km2: (i) on the engine's diet (all energy from
infauna), (ii) on the budget's diet mix (infauna 0.511 of the mass, the rest
at the other pools' qualities), against the budget's 0.029.
The variants change one quantity at a time, then combinations; values are
the literature ranges in summary / tables/literature_evidence.md.
"""
TPY = 1460
EAT, MOVE, REST = 0.957, 0.041, 0.002
FILL, ME = 0.722, 1800.0
P_REAL, P_BUDGET = 0.708, 0.821
B_BEN, B_INF = 1.53, 93.6
SHARE_INF = 0.511
# other budget diet items and their quality (assimilation x E, MJ/t)
OTHER = [(0.209, 0.8 * 3750), (0.084, 0.8 * 7125), (0.019, 0.8 * 5252)]


def need(rm=3.6, c_eat=2.0, c_move=1.6, ec=4500.0, carry=True, sda=0.0,
         q_inf=1820.0, assim=0.7, p=P_REAL):
    """(assimilated MJ needed, ingested-energy factor) per t per yr.

    sda > 0: SDA charged as that fraction of INGESTED energy (instead of a
    per-tick surcharge, then c_eat should be 1.0). Assimilated energy A
    must then cover costs + production + sda x A / assim.
    """
    cost = rm * TPY * (EAT * c_eat + MOVE * c_move + REST * 1.0)
    prod = p * (ec + (FILL * ME if carry else 0.0))
    base = cost + prod
    if sda > 0:
        base = base / (1.0 - sda / assim)
    return base


def report(name, **kw):
    q_inf = kw.get('q_inf', 1820.0)
    n = need(**kw)
    qb = n / q_inf
    m2_engine = qb * B_BEN / B_INF
    q_mix = (SHARE_INF * q_inf + sum(s * q for s, q in OTHER)) / (SHARE_INF + sum(s for s, _ in OTHER))
    qb_mix = n / q_mix
    m2_budget_diet = qb_mix * SHARE_INF * B_BEN / B_INF
    print(f'{name:70} {n:7.0f} {qb:6.2f} {m2_engine:6.3f} {qb_mix:6.2f} {m2_budget_diet:6.3f}')


def main():
    print(f'{"variant":70} {"MJ/t/yr":>7} {"QBinf":>6} {"M2eng":>6} {"QBmix":>6} {"M2mix":>6}')
    report('engine today (feeding 2.0, ec 4500 + carry, 1820 MJ/t infauna)')
    report('feeding_cost 1.0 (eat = routine RM, no SDA at all)', c_eat=1.0)
    report('feeding_cost 1.0 + SDA 0.15 of ingested (intake-proportional)', c_eat=1.0, sda=0.15)
    report('feeding_cost 1.0 + SDA 0.10', c_eat=1.0, sda=0.10)
    report('feeding_cost 1.0 + SDA 0.20', c_eat=1.0, sda=0.20)
    report('feeding_cost 1.5', c_eat=1.5)
    report('RM 3.0 (SMR, no routine factor), feeding 2.0', rm=3.0)
    report('production at lean ec 3600 + carry (whole body 4500 at fill 0.5)', ec=4500 - 0.5 * ME)
    report('production without the reserve carry (ec 4500 only)', carry=False)
    report('infauna quality 2600 x 0.80 = 2080', q_inf=2080.0)
    report('infauna quality 2600 x 0.85 = 2210', q_inf=2210.0)
    report('infauna quality 3000 x 0.80 = 2400 (cropped parts, less ash)', q_inf=2400.0)
    report('budget production 0.821 instead of realised 0.708', p=P_BUDGET)
    print('combinations:')
    report('feeding 1.0 + SDA 0.15 + lean ec', c_eat=1.0, sda=0.15, ec=3600.0)
    report('feeding 1.0 + SDA 0.15 + lean ec + infauna 2080', c_eat=1.0, sda=0.15, ec=3600.0, q_inf=2080.0, assim=0.8)
    report('feeding 1.0 + SDA 0.15 + lean ec + infauna 2400', c_eat=1.0, sda=0.15, ec=3600.0, q_inf=2400.0, assim=0.8)
    print(f'budget: Q/B 3.48, infauna share 0.511 -> M2 {3.48 * SHARE_INF * B_BEN / B_INF:.3f}')


if __name__ == '__main__':
    main()
