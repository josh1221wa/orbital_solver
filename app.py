import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import pandas as pd
import streamlit as st
import io

# Global variables


class OrbitData:
    def __init__(self):
        self.el = np.zeros(10)  # [P, T, e, a, W, w, i, K1, K2, V0]
        self.elerr = np.zeros(10)
        self.fixel = np.ones(10, dtype=int)
        self.elname = ['P', 'T', 'e', 'a', 'W', 'w', 'i', 'K1', 'K2', 'V0']
        self.pos = None
        self.rv1 = None
        self.rv2 = None
        self.obj = {'name': '', 'radeg': 0.0, 'dedeg': 0.0, 'npos': 0, 'nrv1': 0, 'nrv2': 0,
                    'rms': np.zeros(4), 'chi2n': np.zeros(4), 'chi2': 0.0, 'fname': '',
                    'parallax': 0.0}
        self.graph = {'mode': 0}


orb = OrbitData()

# Constants
G = 2945.98  # Gravitational constant in km^3 s^-2 M_sun^-1 day^-1

# Ephemeris calculation


def eph(el, t, rho=False, rv=False):
    n = len(t)
    res = np.zeros((n, 2), dtype=float)
    pi2 = 2 * np.pi
    gr = 180 / np.pi

    P, T, SF, a, W, w, i, K1, K2, V0 = el
    CF2 = 1 - SF**2
    CF = np.sqrt(CF2)
    EC = np.sqrt((1 + SF) / (1 - SF))
    CWW = np.cos(W / gr)
    SWW = np.sin(W / gr)
    CW = np.cos(w / gr)
    SW = np.sin(w / gr)
    CI = np.cos(i / gr)
    SI = np.sin(i / gr)

    if rv:
        for i in range(n):
            dt = t[i] - T
            phase = (dt / P) % 1
            if phase < 0:
                phase += 1
            ANM = phase * pi2
            E = ANM
            E1 = E + (ANM + SF * np.sin(E) - E) / (1 - SF * np.cos(E))
            while abs(E1 - E) > 1e-5:
                E = E1
                E1 = E + (ANM + SF * np.sin(E) - E) / (1 - SF * np.cos(E))
            V = 2 * np.arctan(EC * np.tan(E1 / 2))
            U = V + w / gr
            CU = np.cos(U)
            A1 = SF * CW + CU
            res[i, 0] = V0 + K1 * A1
            res[i, 1] = V0 - K2 * A1
    else:
        AA = a * (CW * CWW - SW * SWW * CI)
        BB = a * (CW * SWW + SW * CWW * CI)
        FF = a * (-SW * CWW - CW * SWW * CI)
        GG = a * (-SW * SWW + CW * CWW * CI)
        for i in range(n):
            dt = t[i] - T
            phase = (dt / P) % 1
            if phase < 0:
                phase += 1
            ANM = phase * pi2
            E = ANM
            E1 = E + (ANM + SF * np.sin(E) - E) / (1 - SF * np.cos(E))
            while abs(E1 - E) > 1e-5:
                E = E1
                E1 = E + (ANM + SF * np.sin(E) - E) / (1 - SF * np.cos(E))
            V = 2 * np.arctan(EC * np.tan(E1 / 2))
            CV = np.cos(V)
            R = CF2 / (1 + SF * CV)
            X = R * CV
            Y = R * np.sin(V)
            res[i, 0] = AA * X + FF * Y
            res[i, 1] = BB * X + GG * Y

    if rho:
        rho_vals = np.sqrt(res[:, 0]**2 + res[:, 1]**2)
        theta = np.arctan2(res[:, 1], res[:, 0]) * 180 / np.pi
        theta = (theta + 360) % 360
        res[:, 0] = theta
        res[:, 1] = rho_vals

    return res

# Coordinate parsing


def getcoord(s):
    l = s.find('.')
    deg = int(s[:l]) if l > 0 else int(s)
    min_part = float(s[l:]) if l > 0 else 0
    res = abs(deg) + min_part
    if float(s) < 0:
        res = -res
    return res

# Time correction


def correct(data, t0):
    time = data[:, 0]
    for i in range(len(time)):
        if time[i] < 3000 and t0 > 3000:
            data[i, 0] = 365.242198781 * (time[i] - 1900) + 15020.31352
        elif time[i] > 3000 and t0 < 3000:
            data[i, 0] = 1900 + (time[i] - 15020.31352) / 365.242198781

# Read input file


def readinp(file_content, fname):
    global orb
    orb.el = np.zeros(10)
    orb.fixel = np.ones(10, dtype=int)
    nmax = 500
    orb.pos = np.zeros((nmax, 6))
    orb.rv1 = np.zeros((nmax, 3))
    orb.rv2 = np.zeros((nmax, 3))
    orb.obj['fname'] = fname

    lines = file_content.decode('utf-8').splitlines()

    kpos = 0
    krv1 = 0
    krv2 = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('C'):
            continue
        fix = 0 if line.startswith('*') else 1
        parts = line.split()
        if not parts:
            continue
        if parts[0] == 'Object:':
            orb.obj['name'] = ' '.join(parts[1:])
        elif parts[0] == 'RA:':
            orb.obj['radeg'] = 15 * getcoord(parts[1])
        elif parts[0] == 'Dec:':
            orb.obj['dedeg'] = getcoord(parts[1])
        elif parts[0] == 'Parallax:':
            orb.obj['parallax'] = float(parts[1])
        elif parts[0] in orb.elname:
            ind = orb.elname.index(parts[0])
            orb.el[ind] = float(parts[1])
            orb.fixel[ind] = fix
        elif 'I1' in line and len(parts) >= 4:
            orb.pos[kpos, 0:4] = [float(p) for p in parts[0:4]]
            kpos += 1
        elif 'Va' in line and len(parts) >= 3:
            orb.rv1[krv1, 0:3] = [float(p) for p in parts[0:3]]
            krv1 += 1
        elif 'Vb' in line and len(parts) >= 3:
            orb.rv2[krv2, 0:3] = [float(p) for p in parts[0:3]]
            krv2 += 1

    orb.pos = orb.pos[:kpos, :] if kpos > 0 else np.array([])
    orb.rv1 = orb.rv1[:krv1, :] if krv1 > 0 else np.array([])
    orb.rv2 = orb.rv2[:krv2, :] if krv2 > 0 else np.array([])

    if kpos > 0:
        correct(orb.pos, orb.el[1])
    if krv1 > 0:
        correct(orb.rv1, orb.el[1])
    if krv2 > 0:
        correct(orb.rv2, orb.el[1])

    st.write(f"Position measures: {kpos}")
    st.write(f"RV measures: {krv1}, {krv2}")
    orb.obj['npos'] = kpos
    orb.obj['nrv1'] = krv1
    orb.obj['nrv2'] = krv2
    orb.elerr = np.zeros(10)
    orb.graph['mode'] = 1 if (krv1 > 0 or krv2 > 0) else 0

# Orbit plotting


def orbplot():
    global orb
    name = orb.obj['fname'].split('.')[0]

    if orb.obj['npos'] > 0:
        fig, ax = plt.subplots(figsize=(6, 6))
        time = np.linspace(0, orb.el[0], 100) + orb.el[1]
        xye = eph(orb.el, time)
        gr = 180 / np.pi
        xobs = -orb.pos[:, 2] * np.sin(orb.pos[:, 1] / gr)
        yobs = orb.pos[:, 2] * np.cos(orb.pos[:, 1] / gr)
        xy0 = eph(orb.el, orb.pos[:, 0])

        ax.plot(-xye[:, 1], xye[:, 0], 'k-', label='Orbit')
        ax.plot(xobs, yobs, 'bs', label='Observations')
        for i in range(len(xobs)):
            ax.plot([xobs[i], -xy0[i, 1]], [yobs[i], xy0[i, 0]], 'k--')
            year = int(round(orb.pos[i, 0]))
            ax.text(xobs[i], yobs[i], str(year), fontsize=8)
        ax.plot([0], [0], 'r*', markersize=10, label='Center')
        ax.set_xlabel('X, arcsec (East)')
        ax.set_ylabel('Y, arcsec (North)')
        ax.set_title(f"Visual Orbit of {orb.obj['name']}")
        ax.axis('equal')
        ax.legend()
        st.pyplot(fig)

    if orb.obj['nrv1'] > 0 or orb.obj['nrv2'] > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        t = np.linspace(min(np.concatenate([orb.rv1[:, 0] if orb.obj['nrv1'] > 0 else np.array([]),
                                           orb.rv2[:, 0] if orb.obj['nrv2'] > 0 else np.array([])])),
                        max(np.concatenate([orb.rv1[:, 0] if orb.obj['nrv1'] > 0 else np.array([]),
                                           orb.rv2[:, 0] if orb.obj['nrv2'] > 0 else np.array([])])),
                        100)
        rv = eph(orb.el, t, rv=True)

        if orb.obj['nrv1'] > 0:
            ax.errorbar(orb.rv1[:, 0], orb.rv1[:, 1],
                        yerr=orb.rv1[:, 2], fmt='bo', label='Primary RV')
        if orb.obj['nrv2'] > 0:
            ax.errorbar(orb.rv2[:, 0], orb.rv2[:, 1],
                        yerr=orb.rv2[:, 2], fmt='ro', label='Secondary RV')

        if orb.obj['nrv1'] > 0:
            ax.plot(t, rv[:, 0], 'b-', label='Primary Fit')
        if orb.obj['nrv2'] > 0:
            ax.plot(t, rv[:, 1], 'r--', label='Secondary Fit')

        ax.set_xlabel('Time (JD)')
        ax.set_ylabel('Radial Velocity (km/s)')
        ax.set_title(f"RV Curve of {orb.obj['name']} vs Time")
        ax.legend()
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        t_phase = np.linspace(0, orb.el[0], 100) + orb.el[1]
        rv = eph(orb.el, t_phase, rv=True)
        phases = np.linspace(0, 1, 100)

        if orb.obj['nrv1'] > 0:
            phase1 = ((orb.rv1[:, 0] - orb.el[1]) / orb.el[0]) % 1
            phase1[phase1 < 0] += 1
            ax.errorbar(
                phase1, orb.rv1[:, 1], yerr=orb.rv1[:, 2], fmt='bo', label='Primary RV')
        if orb.obj['nrv2'] > 0:
            phase2 = ((orb.rv2[:, 0] - orb.el[1]) / orb.el[0]) % 1
            phase2[phase2 < 0] += 1
            ax.errorbar(
                phase2, orb.rv2[:, 1], yerr=orb.rv2[:, 2], fmt='ro', label='Secondary RV')

        if orb.obj['nrv1'] > 0:
            ax.plot(phases, rv[:, 0], 'b-', label='Primary Fit')
        if orb.obj['nrv2'] > 0:
            ax.plot(phases, rv[:, 1], 'r--', label='Secondary Fit')

        ax.set_xlabel('Phase')
        ax.set_ylabel('Radial Velocity (km/s)')
        ax.set_title(f"RV Curve of {orb.obj['name']} vs Phase")
        ax.legend()
        st.pyplot(fig)

# Fit orbital elements


def alleph(params, i):
    global orb
    selfit = np.where(orb.fixel > 0)[0]
    el0 = orb.el.copy()
    el0[selfit] = params
    e = 0.01
    del_vals = [e * el0[0], e * el0[0], e, e * el0[3],
                1, 1, 1, e * el0[7], e * el0[8], e * el0[7]]

    if i < 2 * orb.obj['npos']:
        j = 1 if i >= orb.obj['npos'] else 0
        time = orb.pos[i - j * orb.obj['npos'], 0]
        res = eph(el0, [time], rho=True)[0, j]
        deriv = np.zeros(10)
        for k in range(10):
            if orb.fixel[k] > 0:
                el1 = el0.copy()
                el1[k] += del_vals[k]
                deriv[k] = (eph(el1, [time], rho=True)
                            [0, j] - res) / del_vals[k]
        return np.concatenate([[res], deriv[selfit]])
    elif i < 2 * orb.obj['npos'] + orb.obj['nrv1']:
        time = orb.rv1[i - 2 * orb.obj['npos'], 0]
        res = eph(el0, [time], rv=True)[0, 0]
        deriv = np.zeros(10)
        for k in range(10):
            if orb.fixel[k] > 0:
                el1 = el0.copy()
                el1[k] += del_vals[k]
                deriv[k] = (eph(el1, [time], rv=True)[
                            0, 0] - res) / del_vals[k]
        return np.concatenate([[res], deriv[selfit]])
    elif i < 2 * orb.obj['npos'] + orb.obj['nrv1'] + orb.obj['nrv2']:
        time = orb.rv2[i - 2 * orb.obj['npos'] - orb.obj['nrv1'], 0]
        res = eph(el0, [time], rv=True)[0, 1]
        deriv = np.zeros(10)
        for k in range(10):
            if orb.fixel[k] > 0:
                el1 = el0.copy()
                el1[k] += del_vals[k]
                deriv[k] = (eph(el1, [time], rv=True)[
                            0, 1] - res) / del_vals[k]
        return np.concatenate([[res], deriv[selfit]])
    return np.zeros(len(selfit) + 1)


def fitorb(rms_only=False):
    global orb
    npos = orb.obj['npos']
    nrv1 = orb.obj['nrv1']
    nrv2 = orb.obj['nrv2']
    n = 2 * npos + nrv1 + nrv2
    yy = np.zeros(n)
    err = np.zeros(n)

    if npos > 0:
        yy[:npos] = orb.pos[:, 1]
        err[:npos] = orb.pos[:, 3] * 180 / np.pi / orb.pos[:, 2]
        yy[npos:2*npos] = orb.pos[:, 2]
        err[npos:2*npos] = orb.pos[:, 3]
    if nrv1 > 0:
        yy[2*npos:2*npos+nrv1] = orb.rv1[:, 1]
        err[2*npos:2*npos+nrv1] = orb.rv1[:, 2]
    if nrv2 > 0:
        yy[2*npos+nrv1:] = orb.rv2[:, 1]
        err[2*npos+nrv1:] = orb.rv2[:, 2]

    selfit = np.where(orb.fixel > 0)[0]
    st.write(
        f"Fitting {len(selfit)} elements: {[orb.elname[i] for i in selfit]}")
    st.write(
        f"Total observations: {n} (npos={npos}, nrv1={nrv1}, nrv2={nrv2})")
    par = orb.el[selfit]

    def residuals(params):
        y1 = np.array([alleph(params, i)[0] for i in range(n)])
        return (yy - y1) / err

    if rms_only:
        y1 = np.array([alleph(par, i)[0] for i in range(n)])
    else:
        result = least_squares(residuals, par, method='lm',
                               max_nfev=1000, ftol=1e-10, xtol=1e-10, verbose=2)
        par = result.x
        y1 = yy - result.fun * err
        orb.el[selfit] = par

        n_params = len(selfit)
        dof = n - n_params
        st.write(f"Degrees of freedom: {dof}")
        if dof > 0:
            chi2 = np.sum(result.fun**2)
            reduced_chi2 = chi2 / dof
            st.write(
                f"Chi-squared: {chi2:.4f}, Reduced Chi-squared: {reduced_chi2:.4f}")

            J = np.array([alleph(par, i)[1:] for i in range(n)])
            st.write(f"Jacobian shape: {J.shape}")

            try:
                JTJ = J.T @ J
                st.write(f"JTJ condition number: {np.linalg.cond(JTJ):.2e}")
                cov = np.linalg.inv(JTJ) * reduced_chi2
                errors = np.sqrt(np.diag(cov))
                orb.elerr[selfit] = errors
                st.write("Covariance matrix computed successfully")
            except np.linalg.LinAlgError as e:
                st.write(f"Error computing covariance: {e}")
                st.write("Using approximate errors")
                orb.elerr[selfit] = np.abs(
                    J.T @ result.fun) * np.sqrt(reduced_chi2) / n
        else:
            st.write("Warning: Not enough degrees of freedom for error estimation")
            orb.elerr[selfit] = np.zeros(len(selfit))

    wt = 1 / err**2
    resid2 = (yy - y1)**2 * wt
    nmin = [0, npos, 2*npos, 2*npos+nrv1]
    nmax = [npos, 2*npos, 2*npos+nrv1, n]
    ndat = [nmax[i] - nmin[i] for i in range(4)]
    sd = [np.sum(resid2[nmin[j]:nmax[j]]) if ndat[j]
          > 0 else 0 for j in range(4)]
    wsum = [np.sum(wt[nmin[j]:nmax[j]]) if ndat[j]
            > 0 else 0 for j in range(4)]
    normchi2 = [sd[j] / ndat[j] if ndat[j] > 0 else 0 for j in range(4)]
    wrms = [np.sqrt(sd[j] / wsum[j]) if wsum[j] > 0 else 0 for j in range(4)]

    st.write(f"CHI2/N: {normchi2}")
    st.write(f"RMS (Theta, rho, RV1, RV2): {wrms}")
    st.write("Fitted Parameters and Errors:")
    for i, idx in enumerate(selfit):
        st.write(
            f"{orb.elname[idx]:<5}: {orb.el[idx]:>10.4f} ± {orb.elerr[idx]:.4f}")

    orb.obj['rms'] = wrms
    orb.obj['chi2n'] = normchi2
    if not rms_only:
        orb.obj['chi2'] = np.sum((yy - y1)**2 / err**2)
        orbplot()

    return yy, y1

# Calculate total mass


def calculate_total_mass(P, a, parallax):
    if parallax <= 0:
        return 0.0
    distance_pc = 1000.0 / parallax
    a_au = a * distance_pc
    total_mass = (a_au**3) / (P**2)
    return total_mass

# Calculate spectroscopic masses


def calculate_spectroscopic_masses(P, e, i, K1, K2):
    if K1 == 0 and K2 == 0:
        return 0.0, 0.0, 0.0

    i_rad = np.radians(i)
    sin_i = np.sin(i_rad)
    sin3_i = sin_i**3
    K_sum = K1 + K2

    M12_sin3i = (K_sum**3 * P * (1 - e**2)**(3/2)) / (2 * np.pi * G)

    q = K1 / K2 if K2 != 0 else float('inf')
    if q != float('inf'):
        M_total = M12_sin3i / sin3_i
        M1 = M_total / (1 + q)
        M2 = q * M1
    else:
        M1 = M12_sin3i / sin3_i
        M2 = 0.0

    return M12_sin3i, M1, M2

# Save results


def orbsave():
    global orb
    name = orb.obj['fname'].split('.')[0]
    output = io.StringIO()

    output.write(f"# Object: {orb.obj['name']}\n")
    output.write(f"# RA: {orb.obj['radeg']/15:.6f}\n")
    output.write(f"# Dec: {orb.obj['dedeg']:.6f}\n")
    output.write(f"# Parallax (mas): {orb.obj['parallax']:.6f}\n")

    elements_data = {
        'Parameter': orb.elname,
        'Value': orb.el,
        'Error': orb.elerr,
        'Fixed': orb.fixel
    }
    elements_df = pd.DataFrame(elements_data)
    output.write("\n# Orbital Elements\n")
    elements_df.to_csv(output, index=False)

    if orb.obj['npos'] > 0:
        res = eph(orb.el, orb.pos[:, 0], rho=True)
        pos_data = {
            'Time': orb.pos[:, 0],
            'PA_Obs': orb.pos[:, 1],
            'Rho_Obs': orb.pos[:, 2],
            'Err': orb.pos[:, 3],
            'PA_Fit': res[:, 0],
            'Rho_Fit': res[:, 1]
        }
        pos_df = pd.DataFrame(pos_data)
        output.write("\n# Position Measurements\n")
        pos_df.to_csv(output, index=False)

    if orb.obj['nrv1'] > 0:
        rv1_fit = eph(orb.el, orb.rv1[:, 0], rv=True)[:, 0]
        rv1_data = {
            'Time': orb.rv1[:, 0],
            'RV_Obs': orb.rv1[:, 1],
            'Err': orb.rv1[:, 2],
            'RV_Fit': rv1_fit
        }
        rv1_df = pd.DataFrame(rv1_data)
        output.write("\n# Primary RV Measurements\n")
        rv1_df.to_csv(output, index=False)

    if orb.obj['nrv2'] > 0:
        rv2_fit = eph(orb.el, orb.rv2[:, 0], rv=True)[:, 1]
        rv2_data = {
            'Time': orb.rv2[:, 0],
            'RV_Obs': orb.rv2[:, 1],
            'Err': orb.rv2[:, 2],
            'RV_Fit': rv2_fit
        }
        rv2_df = pd.DataFrame(rv2_data)
        output.write("\n# Secondary RV Measurements\n")
        rv2_df.to_csv(output, index=False)

    total_mass = calculate_total_mass(
        orb.el[0], orb.el[3], orb.obj['parallax'])
    M12_sin3i, M1, M2 = calculate_spectroscopic_masses(
        orb.el[0], orb.el[2], orb.el[6], orb.el[7], orb.el[8])

    stats_data = {
        'Metric': ['CHI2', 'CHI2/N_Theta', 'CHI2/N_Rho', 'CHI2/N_RV1', 'CHI2/N_RV2',
                   'RMS_Theta', 'RMS_Rho', 'RMS_RV1', 'RMS_RV2', 'Parallax_mas',
                   'Total_Mass_Msun', 'M(1+2)_sin3i', 'M1_Msun', 'M2_Msun'],
        'Value': [orb.obj['chi2'], orb.obj['chi2n'][0], orb.obj['chi2n'][1], orb.obj['chi2n'][2],
                  orb.obj['chi2n'][3], orb.obj['rms'][0], orb.obj['rms'][1], orb.obj['rms'][2],
                  orb.obj['rms'][3], orb.obj['parallax'], total_mass, M12_sin3i, M1, M2]
    }
    stats_df = pd.DataFrame(stats_data)
    output.write("\n# Statistics\n")
    stats_df.to_csv(output, index=False)

    st.write(f"Results prepared for {name}")
    if orb.obj['parallax'] > 0:
        st.write(f"Total system mass: {total_mass:.3f} solar masses")
    else:
        st.write("Parallax not provided, cannot calculate total mass")

    st.write("\nSpectroscopic masses:")
    st.write(f"M(1+2)*sin^3(i) = {M12_sin3i:.6f} solar masses")
    st.write(f"M1 = {M1:.6f} solar masses")
    st.write(f"M2 = {M2:.6f} solar masses")

    output_str = output.getvalue()
    st.download_button(
        label="Download Results as CSV",
        data=output_str,
        file_name=f"{name}_output.csv",
        mime="text/csv"
    )

# Streamlit app


def main():
    st.title("Orbit Solver for Visual and Spectroscopic Binaries")
    st.write("This code was prepared using Prof. Andrea Tokovinin's Method for solving orbits of visual and spectroscopic binaries, and it was transformed to Python and implemented in Streamlit by Prof. Mashhoor Al-Wardat (malwardat@sharjah.ac.ae)")

    st.subheader("Upload Input File")
    uploaded_file = st.file_uploader(
        "Choose an input file (e.g., 'FIN379.inp')", type=['inp', 'txt'])

    if uploaded_file is not None:
        st.session_state['file_content'] = uploaded_file.read()
        st.session_state['filename'] = uploaded_file.name
        st.write(f"File uploaded: {uploaded_file.name}")

        if st.button("Process File"):
            readinp(st.session_state['file_content'],
                    st.session_state['filename'])
            if orb.obj['fname']:
                fitorb()
                orbsave()


if __name__ == "__main__":
    main()
