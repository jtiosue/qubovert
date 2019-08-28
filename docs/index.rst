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


Binary Optimization models
==========================

.. automodule:: qubovert
    :members:


Utility methods
===============

Note that the ``utils`` modules will not be imported with ``from qubovert import *``. You must import ``qubovert.utils`` explicitly.


.. automodule:: qubovert.utils
    :members:


Problems defined
================

Note that the ``problems`` modules will not be imported with ``from qubovert import *``. You must import ``qubovert.problems`` explicitly.

For example, you can use the ``SetCover`` class with any of the following.

.. code:: python

    import qubovert
    qubovert.problems.SetCover(...)
    qubovert.problems.np.SetCover(...)
    qubovert.problems.np.covering.SetCover(...)


.. automodule:: qubovert.problems
    :members:


Copyright
=========

Joseph T. Iosue, joe.iosue@yahoo.com.

.. include:: ../LICENSE
