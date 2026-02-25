{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}
    :no-members:
    :no-inherited-members:

{% block classes %}
{% if classes %}
.. rubric:: Classes

.. autosummary::
   :toctree: .
   :template: autosummary/class.rst
   :nosignatures:

{% for item in classes %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block functions %}
{% if functions %}
.. rubric:: Functions

.. autosummary::
   :toctree: .
   :nosignatures:

{% for item in functions %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}