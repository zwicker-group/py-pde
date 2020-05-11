'''
Functions for creating movies of simulation results


.. autosummary::
   :nosignatures:

   movie_scalar
   movie_multiple
   
   
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import functools
import os
import subprocess as sp
import tempfile
import shutil
from typing import Dict, Any

from .plotting import ScalarFieldPlot, ScaleData
from ..storage.base import StorageBase
from ..tools.docstrings import fill_in_docstring



class Movie:
    """ Class for creating movies from matplotlib figures using ffmpeg """

    def __init__(self, width=None, filename=None, verbose=False,
                 framerate=None, image_folder=None):
        self.width = width          # pixel width of the movie
        self.filename = filename    # filename used to save the movie
        self.verbose = verbose      # verbose encoding information?
        self.framerate = framerate  # framerate of the movie
        self.image_folder = image_folder  # folder where images are stored

        # internal variables
        self.recording = False
        self.frame = 0
        self._delete_images = False
        self._start()


    def __del__(self):
        self._end()


    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.filename is not None:
            self.save(self.filename)
        self._end()
        return False


    def _start(self):
        """ initializes the video recording """
        # create temporary directory for the image files of the movie
        if self.image_folder is None:
            self.image_folder = tempfile.mkdtemp(prefix='movie_')
            self._delete_images = True
        self.frame = 0
        self.recording = True


    def _end(self):
        """ clear up temporary things if necessary """
        if self.recording:
            if self._delete_images:
                shutil.rmtree(self.image_folder)
            self.recording = False


    def clear(self):
        """ delete current status and start from scratch """
        self._end()
        self._start()


    def _add_file(self, save_function):
        """
        Adds a file to the current movie
        """
        if not self.recording:
            raise ValueError('Movie is not initialized.')

        save_function("%s/frame_%09d.png" % (self.image_folder, self.frame))
        self.frame += 1


    def add_image(self, image):
        """
        Adds the data of a PIL image as a frame to the current movie.
        """
        self._add_file(image.save)


    def add_figure(self, fig=None):
        """ adds the figure `fig` as a frame to the current movie """
        import matplotlib.pyplot as plt
        if fig is None:
            fig = plt.gcf()

        if self.width is None:
            dpi = None
        else:
            dpi = self.width / fig.get_figwidth()

        # save image
        save_function = functools.partial(fig.savefig, dpi=dpi)
        self._add_file(save_function)


    def save_frames(self, filename_pattern='./frame_%09d.png', frames='all'):
        """ saves the given `frames` as images using the `filename_pattern` """
        if not self.recording:
            raise ValueError('Movie is not initialized.')

        if 'all' == frames:
            frames = range(self.frame)

        for f in frames:
            shutil.copy(
                "%s/frame_%09d.png" % (self.image_folder, f),
                str(filename_pattern) % f
            )


    def save(self, filename=None, extra_args=None):
        """ convert the recorded images to a movie using ffmpeg """
        if filename is None:
            if self.filename is None:
                raise ValueError('`filename` has to be supplied')
            filename = self.filename
        filename = os.path.expanduser(filename)
        
        if not self.recording:
            raise ValueError('Movie is not initialized.')

        # set parameters
        if extra_args is None:
            extra_args = []
        if self.framerate is not None:
            extra_args += ["-r", self.framerate]

        # construct the call to ffmpeg
        # add the `-pix_fmt yuv420p` switch for compatibility reasons
        #     -> http://ffmpeg.org/trac/ffmpeg/wiki/x264EncodingGuide
        args = ["ffmpeg"]
        if extra_args:
            args += extra_args
        args += [
            "-y",  # don't ask questions
            "-f", "image2",  # input format
            "-i", "%s/frame_%%09d.png" % self.image_folder,  # input data
            "-pix_fmt", "yuv420p",  # pixel format for compatibility
            "-b:v", "1024k",  # high bit rate for good quality
            str(filename)  # output file
        ]

        # spawn the subprocess and capture its output
        proc = sp.Popen(args, stdout=sp.PIPE, stderr=sp.PIPE)
        out, err = proc.communicate()

        # do output anyway, when verbosity is requested
        if self.verbose:
            print(out)
            print(err)

        return out, err



@fill_in_docstring
def movie_scalar(storage: StorageBase,
                 filename: str,
                 scale: ScaleData = 'automatic',
                 extras: Dict[str, Any] = None,
                 progress: bool = True,
                 tight: bool = False,
                 show: bool = True) -> None:
    """ produce a movie for a simulation of a scalar field
    
    Args:
        storage (:class:`~pde.storage.base.StorageBase`):
            The storage instance that contains all the data for the movie
        filename (str):
            The filename to which the movie is written. The extension determines
            the format used.
        scale (str, float, tuple of float):
            {ARG_PLOT_SCALE}
        extras (dict, optional):
            Additional functions that are evaluated and shown for each time 
            step. The key of the dictionary is used as a panel title.
        progress (bool):
            Flag determining whether the progress of making the movie is shown.
        tight (bool):
            Whether to call :func:`matplotlib.pyplot.tight_layout`. This affects
            the layout of all plot elements.
        show (bool):
            Flag determining whether images are shown during making the movie
    """
    quantities = [{'title': 'Concentration', 'source': None}]
    if extras:
        for key, value in extras.items():
            quantities.append({'title': key, 'source': value})
    
    # initialize the plot with the first data point
    plot = ScalarFieldPlot.from_storage(storage, quantities=quantities,
                                        scale=scale, tight=tight, show=show)
    # make the full movie
    plot.make_movie(storage, filename, progress=progress)    
   
    
    
@fill_in_docstring
def movie_multiple(storage: StorageBase,
                   filename: str,
                   quantities=None,
                   scale: ScaleData = 'automatic',
                   progress: bool = True) -> None:
    """ produce a movie for a simulation with n components

    Args:
        storage (:class:`~pse.storage.base.StorageBase`):
            The storage instance that contains all the data for the movie
        filename (str):
            The filename to which the movie is written. The extension determines
            the format used.
        quantities:
            {ARG_PLOT_QUANTITIES}
        scale (str, float, tuple of float):
            {ARG_PLOT_SCALE}
        progress (bool):
            Flag determining whether the progress of making the movie is shown.
    """
    plot = ScalarFieldPlot.from_storage(storage, quantities=quantities,
                                        scale=scale)
    plot.make_movie(storage, filename, progress=progress)

