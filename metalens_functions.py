import tensorflow as tf
import numpy as np
import pandas as pd
import solver as s


# Define the aperture function for a circular aperture
def aperture_function(x, y, radius):
    return (x**2 + y**2) <= radius**2


# Function to calculate the Rayleigh-Sommerfeld diffraction
def rayleigh_sommerfeld(x, y, z, aperture_function, wavlen, aperture_radius):
    k = 2 * np.pi / wavlen
    integral = 0
    delta_x = 0.1  # Pixel size
    delta_y = 0.1
    for x_i in np.arange(-aperture_radius, aperture_radius, delta_x):
        for y_i in np.arange(-aperture_radius, aperture_radius, delta_y):
            r = np.sqrt((x - x_i)**2 + (y - y_i)**2 + z**2)
            integral += aperture_function(x_i, y_i, aperture_radius) * np.exp(1j * k * r) / r
    return integral * delta_x * delta_y / (1j * wavlen * z)


# Function to build the Fourier harmonics used in simulations
def build_fourier_harmonics(PQ, N):
    center_x = (PQ - 1) // 2
    center_y = (PQ - 1) // 2

    fourier_harmonics = np.zeros((PQ, PQ, N, N), dtype=complex)

    xx = np.linspace(0, 1, N)
    X, Y = np.meshgrid(xx, xx)

    for jjj in np.arange(0, PQ):
        for iii in np.arange(0, PQ):
            temp_exp = np.exp(-2j * np.pi * (jjj - center_y) * Y) * np.exp( -2j * np.pi * (iii - center_x) * X)
            fourier_harmonics[iii, jjj, :, :] = temp_exp[np.newaxis, np.newaxis, :, :]

    return fourier_harmonics

def rectangular_pillars_library (fourier_harmonics, params, min_size, min_width, max_width):
    
    steps = int((max_width - min_width) // 0.1 + 2)

    width_values = np.linspace(min_width,max_width,steps)

    transmitted_E_x = np.zeros((steps, params['Nx'], params['Nx']), dtype=complex)
    transmitted_power = np.zeros((steps, params['Nx'], params['Nx']), dtype=complex)

    for width, step in zip(width_values, np.arange(0, steps)):

        pillar_x = np.ones((1,params['pixelsX'],params['pixelsY'],1), dtype = float)
        pillar_y = np.ones((1,params['pixelsX'],params['pixelsY'],1), dtype = float)

        pillar_x[0,0,0,0] = 0.5 * (min_size + width * 1e-06)/params['Lx']
        pillar_y[0,0,0,0] = 0.5 * (min_size + width* 1e-06)/params['Lx']

        r_x = tf.convert_to_tensor(pillar_x, dtype=tf.float32)
        r_y = tf.convert_to_tensor(pillar_x, dtype=tf.float32)

        ER_t, UR_t = s.generate_rectangular_resonators(r_x, r_y, params)

        outputs = s.simulate(ER_t, UR_t, params)

        tX = np.reshape(np.squeeze((outputs['tx'].numpy())), params['PQ'])
        teX = np.sum(tX[:, :, np.newaxis, np.newaxis] * fourier_harmonics, axis=(0,1))

        TX = np.reshape(np.squeeze((outputs['T'].numpy())), params['PQ'])
        TEX = np.sum(TX[:, :, np.newaxis, np.newaxis] * fourier_harmonics, axis=(0,1))

        transmitted_E_x[step, :, :] = teX
        transmitted_power[step,:,:] = TEX

    #average_field = np.mean(transmitted_E_x, axis=(1,2))
    average_power = np.mean(transmitted_power, axis = (1,2))


    return transmitted_E_x, average_power


# Function to generate and simulate rectangular pillars with variable height (loading effect)
def rectangular_pillars_library_variable_height (fourier_harmonics, params, min_size, min_width, max_width, h):
    
    steps = int((max_width - min_width) // 0.1 + 2)

    width_values = np.linspace(min_width,max_width,steps)

    transmitted_E_x = np.zeros((steps, params['Nx'], params['Nx']), dtype=complex)
    transmitted_E_y = np.zeros((steps, params['Nx'], params['Nx']), dtype=complex)
    
    tX_values = []

    transmitted_power = np.zeros((steps, params['Nx'], params['Nx']), dtype=complex)

    L_values = []

    for width, step in zip(width_values, np.arange(0, steps)):

        #The height of the pillar depends on the width (step), loading effect 20%
        L = [h - (h * 0.2 * ((step) /steps)), 1 + (h * 0.2 * ((step)/steps))]
        L_values.append(L[0])

        L = tf.convert_to_tensor(L, dtype = tf.complex64)
        L = L[tf.newaxis, tf.newaxis, tf.newaxis, :, tf.newaxis, tf.newaxis]
        params['L'] = L * params['nanometers'] 

        pillar_x = np.ones((1,params['pixelsX'],params['pixelsY'],1), dtype = float)
        pillar_y = np.ones((1,params['pixelsX'],params['pixelsY'],1), dtype = float)

        pillar_x[0,0,0,0] = 0.5 * (min_size + width * 1e-06)/params['Lx']
        pillar_y[0,0,0,0] = 0.5 * (min_size + width * 1e-06)/params['Lx']

        r_x = tf.convert_to_tensor(pillar_x, dtype=tf.float32)
        r_y = tf.convert_to_tensor(pillar_x, dtype=tf.float32)

        ER_t, UR_t = s.generate_rectangular_resonators(r_x, r_y, params)

        outputs = s.simulate(ER_t, UR_t, params)

        tX = np.reshape(np.squeeze((outputs['tx'].numpy())), params['PQ'])
        tY = np.reshape(np.squeeze((outputs['ty'].numpy())), params['PQ'])
        tZ = np.reshape(np.squeeze((outputs['tz'].numpy())), params['PQ'])

        teX = np.sum(tX[:, :, np.newaxis, np.newaxis] * fourier_harmonics, axis=(0,1))
        teY = np.sum(tY[:, :, np.newaxis, np.newaxis] * fourier_harmonics, axis=(0,1))
        teZ = np.sum(tZ[:, :, np.newaxis, np.newaxis] * fourier_harmonics, axis=(0,1))

        TX = np.reshape(np.squeeze((outputs['T'].numpy())), params['PQ'])
        
        TEX = np.sum(TX[:, :, np.newaxis, np.newaxis] * fourier_harmonics, axis=(0,1))

        transmitted_E_x[step, :, :] = teX
        transmitted_E_y[step, :, :] = teY
        transmitted_power[step,:,:] = TEX

        tX_values.append(tX)


    average_field = np.mean(transmitted_E_x, axis=(1,2))
    average_power = np.mean(transmitted_power, axis = (1,2))

    return transmitted_E_x, average_power, L_values, transmitted_E_y, tX_values


def cross_pillars_library (fourier_harmonics, params, min_size, min_width, max_width, min_length, max_length):
    
    steps_w = int((max_width - min_width) // 0.5 + 2)
    steps_l = int((max_length - min_length) // 0.5 + 2)

    width_values = np.linspace(min_width,max_width,steps_w)
    length_values = np.linspace(min_length,max_length,steps_l)

    transmitted_E_x = np.zeros((steps_w, steps_l, params['Nx'], params['Nx']), dtype=complex)
    transmitted_E_y = np.zeros((steps_w, steps_l, params['Nx'], params['Nx']), dtype=complex)
    transmitted_E_z = np.zeros((steps_w, steps_l, params['Nx'], params['Nx']), dtype=complex)
    
    transmitted_power = np.zeros((steps_w, steps_l, params['Nx'], params['Nx']), dtype=complex)
    
    for length, step_l in zip(length_values, np.arange(0, steps_l)):
        for width, step_w in zip(width_values, np.arange(0, steps_w)):

            pillar_w = np.ones((1,params['pixelsX'],params['pixelsY'],1), dtype = float)
            pillar_l = np.ones((1,params['pixelsX'],params['pixelsY'],1), dtype = float)

            pillar_w[0,0,0,0] = 0.5 * (min_size + width * 1e-06)/params['Lx']
            pillar_l[0,0,0,0] = 0.5 * (min_size + length * 1e-06)/params['Lx']

            w = tf.convert_to_tensor(pillar_w, dtype=tf.float32)
            l = tf.convert_to_tensor(pillar_l, dtype=tf.float32)

            ER_t, UR_t = s.generate_rectangular_resonators(w, l, params)

            outputs = s.simulate(ER_t, UR_t, params)

            tX = np.reshape(np.squeeze((outputs['tx'].numpy())), params['PQ'])
            tY = np.reshape(np.squeeze((outputs['ty'].numpy())), params['PQ'])
            tZ = np.reshape(np.squeeze((outputs['tz'].numpy())), params['PQ'])

            teX = np.sum(tX[:, :, np.newaxis, np.newaxis] * fourier_harmonics, axis=(0,1))
            teY = np.sum(tX[:, :, np.newaxis, np.newaxis] * fourier_harmonics, axis=(0,1))
            teZ = np.sum(tX[:, :, np.newaxis, np.newaxis] * fourier_harmonics, axis=(0,1))

            TX = np.reshape(np.squeeze((outputs['T'].numpy())), params['PQ'])
            
            TEX = np.sum(TX[:, :, np.newaxis, np.newaxis] * fourier_harmonics, axis=(0,1))

            transmitted_E_x[step_w, step_l,:, :] = teX
            transmitted_E_y[step_w, step_l,:, :] = teY
            transmitted_E_z[step_w, step_l,:, :] = teZ


            transmitted_power[step_w, step_l,:,:] = TEX
        

    average_field = np.mean(transmitted_E_x, axis=(1,2))
    average_power = np.mean(transmitted_power, axis = (1,2))

    phase = np.angle(average_field)

    return transmitted_E_x, average_power, transmitted_E_y



def fresnel_propagation(params, near_field, dx, distance_min, distance_max, Nz):

        N, _ = np.shape(near_field)
        xx = dx * np.arange(-N, N)
        X, Y = np.meshgrid(xx, xx)
        zz = np.linspace(distance_min, distance_max, Nz)

        near_field_padded = np.zeros((2 * N, 2 * N), dtype=complex)
        near_field_padded[0:N, 0:N] = near_field
        near_field_fft2 = np.fft.fft2(near_field_padded)

        propagated_field = np.zeros((N, N, Nz), dtype=complex)

        for iii in np.arange(0, Nz):
            r = np.sqrt(X**2 + Y**2 + zz[iii]**2)
            propagator = zz[iii] * np.exp(1j * (2 * np.pi / params['wavelengths']) * r) / (r**2)
            factor = (1 / dx**2) * (1 / (1j * params['wavelengths']))
            propagated_field[:, :, iii] = factor * dx**2 * np.fft.fftshift(np.fft.ifft2(near_field_fft2 * np.fft.fft2(propagator)))[0:N, 0:N]

        return propagated_field

