import numpy as np
from sklearn.decomposition import PCA
import matplotlib as mpl
from matplotlib import pyplot as plt


class TrainingDataset:

    def __init__(self, filename, pca_component_number=5, degrees=7, training_subset=0.7):
        """
        Decompose a saved training dataset using a PCA.

        Parameters
        ----------
        filename: str
           The filename of the stored dataset
        pca_component_number:
           The number of pca components to use
        degrees:
           The degrees of the polynomials to fit.
        training_subset:
           The percentage of the input data to used for training. All data will be used for finding error. [0 - 1]
        """
        self.filename = filename
        self.pca_component_number = pca_component_number
        self.degrees = degrees
        self.training_subset = training_subset

        self.read_training_dataset()
        self.decompose_dataset()

    def read_training_dataset(self):
        """ Reads a training dataset.
        """
        raw_data = np.loadtxt(self.filename, delimiter=',')
        self.y = raw_data[1:, 0]
        self.xi = raw_data[0, 1:]
        self.X = raw_data[1:, 1:]
        self.split_training_subset()

    def split_training_subset(self):
        """ Create a training subset, taking into account only a part of the input data."""
        total_samples = int(len(self.y))
        training_samples = total_samples * self.training_subset

        idxs = np.random.choice(np.arange(total_samples), training_samples, replace=False)
        self.y_train = self.y[idxs]
        self.X_train = self.X[idxs, :]

    def decompose_dataset(self):
        """ Run a PCA to reduce the dimension of the dataset.
        """
        # Run the PCA
        pca = PCA(n_components=self.pca_component_number)
        self.pca_weights = pca.fit_transform(self.X_train)

        # Fit the polynomials for each PCA component
        polynomial_coefficients = self.fit_polynomials()

        # Calculate the matrix
        self.M = pca.components_.T.dot(polynomial_coefficients[:, ::-1])

        self.pca_mean = pca.mean_
        self.pca_components = pca.components_

    def fit_polynomials(self):
        """
        Fit a polynomial to each column of the matrix pca_weights.

        Returns
        -------
        pol_coefficients:
           A matrix with polynomial coefficients
        """
        p_coefficients = []
        for pca_component in range(self.pca_component_number):
            p = np.polyfit(self.y_train, self.pca_weights[:, pca_component], self.degrees)
            p_coefficients.append(p)
        p_coefficients = np.array(p_coefficients)
        return p_coefficients

    def save_approximation(self, filename):
        """
        Saves the approximation in a text file.

        Parameters
        ----------
        filename: str
           Filename of output file.
        """
        data = np.vstack([self.xi, self.pca_mean, self.M.T]).T

        print "Saving results in %s" % filename
        header_txt = "PCA matrix for RB scattering.\n  \t\txi\t\t\t\t\t\tmean\t\t\t\tMatrix [%sx%s]" % (self.M.shape[0], self.M.shape[1])
        np.savetxt(filename, data, header=header_txt)

    def reconstruct_spectrum(self, y):
        """
        Calculate the spectrum for the given parameter y.

        Parameters
        ----------
        y: float
           Parameter x

        Returns
        -------
        X: array
           Rayleigh-Brillouin scattering spectrum.
        """
        # Calculate the powers of y
        y_vector = np.array([y ** n for n in range(self.degrees + 1)])

        # Calculate the distribution difference
        X_diff = self.M.dot(y_vector)

        # Add the PCA mean
        X = X_diff.T + self.pca_mean
        return X

    def max_error(self):
        """ Find the y value that has the maximum error.
        """
        errors = []
        for idx in range(len(self.y)):
            spectrum_error = self.error_for_profile(idx)
            errors.append(max(spectrum_error))

        max_error = max(errors)
        max_idx = errors.index(max_error)
        max_error_y = self.y[max_idx]

        return max_error, max_error_y, max_idx

    def error_for_profile(self, idx):
        y = self.y[idx]
        true_spectrum = self.X[idx, :]
        reconstructed_spectrum = self.reconstruct_spectrum(y)

        spectrum_error = (true_spectrum - reconstructed_spectrum) / np.max(true_spectrum)
        return spectrum_error

    def unexplained_variance_percentage(self, max_components):
        """
        Return the percentage of explained variance for the given number of pca components.

        Parameters
        ----------
        max_components:
           The number of component used for the PCA.

        Returns
        -------
        unexplained_variance:
           The percentage of variance that remains unexplained by the specified number of PCA components.
        """
        pca = PCA(n_components=max_components)  # Define the pca
        pca.fit(self.X)  # Fit the data

        explained_variance = np.cumsum(pca.explained_variance_ratio_) * 100

        unexplained_variance = 100 - explained_variance
        return unexplained_variance

    def plot_unexplained_variance(self, max_components=10, figsize=(3.54, 2.2)):
        # Run the PCA with many components
        unexplained_variance = self.unexplained_variance_percentage(max_components)

        component_no = len(unexplained_variance)
        component_idx = np.arange(component_no) + 1

        # Check the explained variance of each component
        plt.figure(figsize=figsize)
        ax = plt.subplot(111)

        line = plt.plot(component_idx,  unexplained_variance, lw=0.5)[0]

        plt.plot(component_idx, unexplained_variance, '.', color=line.get_color(), ms=10)

        plt.xlabel('No. of PCA components')
        plt.ylabel('Unexplained variance [%]')
        plt.yscale('log')
        plt.minorticks_off()

        ax.get_yaxis().set_major_locator(mpl.ticker.LogLocator(numticks=6))
        plt.draw()


        ymin, ymax = plt.ylim()
        plt.ylim(ymin, 100)

        plt.tight_layout()

    def plot_training_profiles(self, idxs = [0, -1], figsize=(3.54, 2.2)):
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111)

        for n, idx in enumerate(idxs):
            self.draw_training_profile(ax, idx)

        plt.title('Example of training profiles')
        plt.legend()
        plt.tight_layout()
        return fig

    def plot_extreme_y_training_profiles(self, title=None, figsize=(3.54, 2.2)):
        """
        Plot the RB spectra for the extreme values of y in the training dataset.

        Parameters
        ----------
        title: str or None
           The title of the plot. If None, the default title is used. To have no title
           use the blank string i.e. "".

        Returns
        -------
        fig:
           The figure object.
        """
        y_min_idx, y_max_idx = self._get_extreme_y_idx()
        fig = self.plot_training_profiles(idxs=[y_min_idx, y_max_idx], figsize=figsize)

        if title is None:
            title = 'Training profiles for extreme y values'
        plt.title(title)

        return fig

    def _get_extreme_y_idx(self):
        y_min_idx = np.argmin(self.y)
        y_max_idx = np.argmax(self.y)
        return y_min_idx, y_max_idx

    def draw_training_profile(self, ax, idx):
        profile = self.X[idx, :]
        plt.plot(self.xi, profile, label='y=%.3f' % self.y[idx])
        plt.ylabel('Intensity [a.u.]')
        plt.xlabel(r'x')
        ax.locator_params(nbins=5)
        plt.xlim(self.xi[0], self.xi[-1])

    def plot_explained_variance(self, max_components=10, figsize=(3.54, 2.2)):
        # Run the PCA with many components
        pca = PCA(n_components=max_components)  # Define the pca
        pca.fit(self.X)  # Fit the data

        component_no = len(pca.explained_variance_)
        component_idx = np.arange(component_no) + 1

        # Check the explained variance of each component
        plt.figure(figsize=figsize)
        line = plt.plot(component_idx, pca.explained_variance_ratio_ * 100,  lw=0.5)[0]
        plt.plot(component_idx, pca.explained_variance_ratio_ * 100, '.', color=line.get_color(), ms=10)

        plt.xlabel('No. of PCA components')
        plt.ylabel('Explained variance [%]')
        plt.yscale('log')

        ymin, ymax = plt.ylim()
        plt.ylim(ymin, 100)

        plt.tight_layout()

    def plot_decomposition(self, components=2, figsize=(3.54, 2.2)):
        # Plot the components
        plt.figure(figsize=figsize)
        ax1 = plt.subplot(211)
        self.draw_pca_mean(ax1)

        ax2 = plt.subplot(212, sharex=ax1)
        self.draw_components(ax2, components)

        plt.tight_layout()

    def plot_components(self, components=3, figsize=(3.54, 2.2)):
        plt.figure(figsize=figsize)
        ax1 = plt.subplot(111)
        self.draw_components(ax1, components, ylabel=True)
        plt.tight_layout()

    def draw_pca_mean(self, ax):
        plt.title('Mean distribution')
        plt.ylabel('Intensity [a.u.]')
        plt.xlabel(r'x')
        plt.plot(self.xi, self.pca_mean)
        plt.xlim(self.xi[0], self.xi[-1])
        ax.locator_params(nbins=5, axis='y')

    def draw_components(self, ax, components=3, ylabel=False):

        shown_components = min(components, self.pca_component_number)

        plt.title('Base functions')

        for c in range(shown_components):
            plt.plot(self.xi, self.pca_components[c, :], label='Base %s' % (c + 1))

        plt.xlabel(r'x')

        if ylabel:
            plt.ylabel('Intensity [a.u.]')

        plt.xlim(self.xi[0], self.xi[-1])
        plt.legend()
        ax.locator_params(nbins=5, axis='y')

    def plot_polynomials(self, degree=None, extrapolation=0, axes_per_row=4, fig_width=7.48, subplot_height=2.2):

        if degree is None:
            degree = self.degrees

        # Extrapolate the polynomials at 1/5 of the y range
        min_y = min(self.y_train)
        max_y = max(self.y_train)
        extrapolation_length = (max_y - min_y) * extrapolation

        y_grid_min = min_y - extrapolation_length
        y_grid_max = max_y + extrapolation_length

        y_grid = np.linspace(y_grid_min, y_grid_max, 100)  # Grid for calculate the polynomial

        # Calculate number of rows (5 plots per row)
        number_of_rows = (self.pca_component_number - 1) / axes_per_row + 1

        plt.figure(figsize=(fig_width, subplot_height * number_of_rows))

        for n_component in range(self.pca_component_number):  # For each component
            p = np.polyfit(self.y_train, self.pca_weights[:, n_component], degree)  # Fit a polynomial

            # Subplot
            w_p = np.polyval(p, y_grid)  # Calculate the polynomial on a grid

            ax = plt.subplot(number_of_rows, axes_per_row, n_component + 1)
            plt.plot(y_grid, w_p, lw=3)
            plt.plot(self.y_train, self.pca_weights[:, n_component], '.', ms=2)
            plt.title('PC %s' % (n_component + 1))
            plt.xlabel('y')
            ax.locator_params(nbins=4, axis='x', tight=True)
            ax.locator_params(nbins=5, axis='y', tight=False)

            if n_component % axes_per_row == 0:  # Only for the first plot per row
                plt.ylabel('Weights')

        plt.tight_layout()

    def plot_reconstructed_spectrum(self, y, figsize=(3.54, 2.2)):
        """ Plot spectrum for the given y parameter
        """
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111)
        self.draw_reconstructed_spectrum(ax, y)

        plt.tight_layout()
        return fig, ax

    def draw_reconstructed_spectrum(self, ax, y):
        spectrum = self.reconstruct_spectrum(y)
        plt.plot(self.xi, spectrum)
        plt.xlabel(r'x')
        plt.ylabel('Intensity [a.u.]')
        ax.locator_params(nbins=5, axis='y')

    def plot_max_error_spectrums(self, figsize=(3.54, 2.2)):

        _, y, max_idx = self.max_error()
        self.plot_profile_comparison(max_idx, top_title="Spectrum for y = %.3f" % y, figsize=figsize)

    def plot_profile_comparison(self, idx, top_title=None, figsize=(3.54, 2.2)):
        """ Plot spectrum for the given y parameter
        """
        fig = plt.figure(figsize=figsize)

        y = self.y[idx]
        true_spectrum = self.X[idx, :]
        reconstructed_spectrum = self.reconstruct_spectrum(y)
        error = self.error_for_profile(idx)

        ax1 = plt.subplot(211)
        plt.plot(self.xi, true_spectrum, label='True')
        plt.plot(self.xi, reconstructed_spectrum, label='Reconstructed')
        plt.ylabel('Intensity [a.u.]')
        plt.xlim(self.xi[0], self.xi[-1])
        plt.legend()
        ax1.locator_params(nbins=5, axis='y')
        plt.title(top_title)

        ax2 = plt.subplot(212, sharex=ax1)
        plt.plot(self.xi, error * 100)

        plt.xlim(self.xi[0], self.xi[-1])
        plt.xlabel(r'x')
        plt.ylabel('Error [%]')
        plt.title('Error relative to peak')
        ax2.locator_params(nbins=5, axis='y')

        plt.tight_layout()
        return fig, ax1


class StoredApproximation:

    def __init__(self, filename):
        self.filename = filename
        self.read_stored_approximation()

    def read_stored_approximation(self):
        """ Load a stored approximation from a file. """
        data = np.loadtxt(self.filename)
        self.xi = data[:, 0]
        self.pca_mean = data[:,1]
        self.M = data[:,2:]

    def spectrum(self, y):
        """
        Calculate the spectrum for the given parameter y.

        Parameters
        ----------
        y: float
           Parameter x

        Returns
        -------
        X: array
           Rayleigh-Brillouin scattering spectrum.
        """
        # Degree of polynomial
        p_degree = self.M.shape[1]

        # Calculate the powers of y
        y_vector = np.array([y**n for n in range(p_degree)])

        # Calculate the distribution difference
        X_diff = self.M.dot(y_vector)

        # Add the PCA mean
        X = X_diff.T + self.pca_mean

        return X

    def plot_spectrum(self, y):
        """ Plot spectrum for the given y parameter
        """
        fig = plt.figure()
        ax = plt.subplot(111)
        self.draw_spectrum(ax, y)

        plt.tight_layout()
        return fig, ax

    def draw_spectrum(self, ax, y):
        spectrum = self.spectrum(y)
        plt.plot(self.xi, spectrum)
        plt.xlabel(r'x')
        plt.ylabel('Intensity [a.u.]')

