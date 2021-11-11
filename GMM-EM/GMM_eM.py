import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import random
from mpl_toolkits.mplot3d import Axes3D


csvfile = 'points.csv'

def read_csv(csvf): #csv 파일 읽기, csv file read
    data = np.genfromtxt(open(csvf, "rb"),dtype=float, delimiter=",", skip_header=0)
    return data

def initiate():
    #pis = [0.27232708, 0.27546473, 0.45220819]
    #mus = np.array([[69.20722901, 19.70652909],
   # [29.84835715, 80.14776865],
    #[79.80717341, 69.65192526]])
    #sigmas = [[85.2867814 ,  3.3551158 ], [ 3.3551158 , 84.26381303]], [[115.31568863 ,  0.72129237],
    #[0.72129237 , 85.71009274]], [[216.90027901, 129.13394185], [129.13394185 ,211.39333946]]
    pis = [1 / 4, 1 / 4, 1 / 2]
    mus = np.array([[123, 121], [23, 63], [21, 31]])
    sigmas = [[88, 63], [33, 34]], [[131, 23.72],
     [0.7212 , 52.71009274]], [[202, 111.13394185], [126.133 ,200.39333946]]

    return pis, mus, sigmas


def solve(data, max_iter, pi, mu, cov):

    print("starting with:")
    print("pis = ", pi)
    print("mus = \n", mu)
    print("covs = \n", cov)
    print()

    plot_data(mu, cov, pi)

    converged = False
    wait = 0
    for it in range(max_iter):
        if not converged:
            """E-Step"""
            r, m_c, pi = e_step(data, mu, cov, pi)

            """M-Step"""
            mu0, cov = m_step(data, r, m_c)

            print("iteration", it, ":")
            print("pis = ", pi)
            print("mus = \n", mu0)
            print("cov = \n", cov)
            print()
            if it % 2 == 0:
                plot_data(mu, cov, pi)

            """convergence condition"""
            shift = np.linalg.norm(np.array(mu) - np.array(mu0))
            mu = mu0
            if shift < 0.0001:
                wait += 1
                if wait > 4:
                    converged = True
            else:
                wait = 0
    final_result(data, pi, mu, cov)


def e_step(data, mu, cov, pi):
    clusters_number = len(pi)

    """creating estimations gaussian density functions"""
    gaussian_pdf_list = []
    for j in range(clusters_number):
        gaussian_pdf_list.append(multivariate_normal(mu[j], cov[j]))

    """Create the array r with dimensionality nxK"""
    r = np.zeros((len(data), clusters_number))

    """Probability for each data point x_i to belong to gaussian g """
    for c, g, p in zip(range(clusters_number), gaussian_pdf_list, pi):
        r[:, c] = p * g.pdf(data)

    """Normalize the probabilities 
    each row of r sums to 1 and weight it by mu_c == the fraction of points belonging to cluster c"""
    for i in range(len(r)):
        sum1 = np.dot([1, 1, 1], r[i, :].reshape(clusters_number, 1))
        r[i] = r[i] / sum1

    """calculate m_c
    For each cluster c, calculate the m_c and add it to the list m_c"""
    m_c = []
    for c in range(clusters_number):
        m = np.sum(r[:, c])
        m_c.append(m)

    """calculate pi
    probability of occurrence for each cluster"""
    for k in range(clusters_number):
        pi[k] = (m_c[k] / np.sum(m_c))

    return r, m_c, pi


def m_step(data, r, m_c):
    clusters_number = len(m_c)

    mu = []
    """calculate mu"""
    for k in range(clusters_number):
        mu.append(np.dot(r[:, k].reshape(len(data)), data) / m_c[k])
    mu = np.array(mu)

    cov = []
    """calculate sigma"""
    for c in range(clusters_number):
        dr = np.stack((r[:, c], r[:, c]), axis=-1)
        temp = (dr * (data - mu[c])).T @ (data - mu[c])
        cov.append(temp / m_c[c])
    cov = np.array(cov)

    return mu, cov


def plot_data(mu, cov, p):
    X, Y, gaussians = drawable_gaussian(p, mu, cov)

    # Make a 3D plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, gaussians, cmap='viridis', linewidth=0)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()


def final_result(samples, p, mu, cov):
    X, Y, gaussians = drawable_gaussian(p, mu, cov)

    fig, ax = plt.subplots()

    CS = ax.contour(X, Y, gaussians)
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_title('contour of Gaussian distributions and the data')

    new_list1 = np.array(random.sample(list(samples), 500))
    plt.plot(new_list1[:, 0], new_list1[:, 1], 'x')

    plt.axis('equal')
    plt.show()


def drawable_gaussian(p, mu, cov):
    x = np.arange(0, 150, 1)
    y = np.arange(0, 150, 1)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    gaussians = 0
    for i in range(len(p)):
        gaussians += p[i] * multivariate_normal(mu[i], cov[i]).pdf(pos)
    return X, Y, gaussians


def main():
    """generate samples from different distributions"""
    mu0 = [[69.20722901, 19.70652909],[29.84835715, 80.14776865],[79.80717341, 69.65192526]]
    sig0 = [[85.2867814 ,  3.3551158 ], [ 3.3551158 , 84.26381303]], [[115.31568863 ,  0.72129237],
    [0.72129237 , 85.71009274]], [[216.90027901, 129.13394185], [129.13394185 ,211.39333946]]
    samples = read_csv(csvfile)

    """initialize pi mu and sigma of distributions"""
    pis, mus, sigmas = initiate()

    """solve the problem using EM algorithm"""
    solve(samples, 200, pis, mus, sigmas)


if __name__ == '__main__':
    main()