Welcome to qubovert's documentation!
====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


README
======

.. include:: ../README.rst


Problems defined
================

Note that these are all imported to use globally. For example, you can use the ``SetCover`` class with any of the following.

.. code:: python

    import qubovert
    qubovert.SetCover(...)
    qubovert.problems.SetCover(...)
    qubovert.problems.np.SetCover(...)
    qubovert.problems.np.covering.SetCover(...)


.. automodule:: qubovert.problems
    :members:


Utility methods
===============

Note that the ``utils`` modules will not be imported with ``from qubovert import *``. You must import ``qubovert.utils`` explicitly.


.. automodule:: qubovert.utils
    :members:


Copyright
=========

Joseph T. Iosue, joe.iosue@yahoo.com.

.. include:: ../LICENSE
