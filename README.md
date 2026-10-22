# Sensora

## Run Online

Sensora is hosted on Streamlit Community Cloud: https://sensora-web.streamlit.app/

You give it a pressure range, and it searches for the soft polymer matrix, ceramic fillers and
loadings that give a capacitive pressure sensor the highest robust capacitance sensitivity over
that range. The app has three pages:

- **Composite Optimizer:** runs the genetic-algorithm search and shows ranked formulations,
  predicted response curves and an experiment plan.
- **Database:** the matrices and fillers the optimizer can choose from, which you can add to or delete from.
- **AI Assistant:** answers questions about the sensors, materials and model from Sensora's own knowledge base.

## Run Locally

Requires Python 3.11 or newer.

```bash
git clone https://github.com/SRD-NTU/Sensora.git
cd Sensora
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501 in your browser.

The app requires Supabase for material storage. Configure `SUPABASE_URL` and
`SUPABASE_SECRET_KEY` as described in [supabase/README.md](supabase/README.md).
The AI Assistant page can download a small language model from within the app.


## Motivation and Problem Statement
Imagine building a robotic hand that can sense how firmly it holds an object, a wearable device that monitors subtle body movements, or a sensor that detects pressure changes in a structure. In all these applications, the sensing layer plays a crucial role. But designing one is not as simple as choosing a material and making a sensor. A researcher looking for a suitable dielectric layer faces hundreds of possible polymers, fillers, and material combinations reported across the literature. Some promise excellent sensitivity but require complicated fabrication techniques, expensive materials, or specialized equipment. Others may be easier to fabricate but perform poorly within the required pressure range. The challenge is even greater for researchers outside materials science. Why should someone developing a robotic system have to spend months experimenting with dielectric composites just to build a pressure sensor that meets their needs? More importantly, with so much existing knowledge about these materials, can we use it to make sensor design more accessible, predictable, and less dependent on trial and error? 

This is the idea behind Sensora, an AI-powered platform that helps researchers design dielectric layers based on what they actually need and what materials they have available. 
## Framework
Instead of starting with a long literature search and repeated experiments, users can select available polymers, such as PDMS or Ecoflex, choose suitable dielectric fillers, define their acceptable filler concentrations, and specify the pressure range in which their sensor should operate. Behind the scenes, Sensora combines a materials database with physics-based models that estimate how different compositions affect dielectric properties and mechanical compliance. A genetic algorithm then explores possible formulations and identifies promising combinations that could deliver higher pressure sensitivity within the user's constraints. The platform also shows predicted sensing responses and material properties, allowing users to understand the trade-offs before moving to fabrication. Sensora is not intended to replace experiments. It is designed to make the experiments smarter.

## Genetic Algorithm Based Design of Materials for Capacitive Pressure Sensors

A genetic algorithm (GA) is a population-based optimization method that evolves candidate solutions (chromosomes) through fitness-based selection, crossover, and mutation. In Sensora, each chromosome encodes the polymer-matrix composition, filler identities, and their volume fractions. Regime-specific physics models estimate the effective dielectric and elastic properties of each candidate, while a planar-capacitor model predicts its pressure-dependent response. The GA iteratively optimizes these formulations over successive generations based on their predicted performance and feasibility constraints. The optimization problem is formulated as follows:


$$
\max_{\mathbf{x}} F(\mathbf{x})
=
\varepsilon_{r,\mathrm{rob}}(\mathbf{x})
\frac{
r_{\mathrm{rob}}(P_{\max};\mathbf{x})
-
r_{\mathrm{rob}}(P_{\min};\mathbf{x})
}{
P_{\max}-P_{\min}
},
\qquad
\text{subject to}
\qquad
\frac{1000P_{\max}}{E_{\mathrm{soft}}(\mathbf{x})}
\leq 0.10
$$

Where $r_{rob}$ is its conservative normalized capacitance response, pressures P are in kPa, and $E_{soft}$ is in Pa. Starting from random candidates, the GA retains strong solutions, selects parents by tournament, recombines and mutates their genes, and repairs invalid compositions.  Formulations that exceed the 10% strain limit receive a large negative fitness, so only feasible designs appear in the final ranking.

## Key Contribution
Sensora Web Application. The app is hosted [here](https://sensora-web.streamlit.app/) and features three components. 

**Composite Optimizer.** Users specify a pressure range, select eligible polymer matrices and ceramic fillers, and choose filler-loading limits Sensora ranks feasible composite designs. 

**Materials Database.** Users can customize the materials database by adding or removing polymer matrices and ceramic fillers.  

**AI Assistant**. Sensora also features a conversational interface that enables users to explore and interact with Sensora’s knowledge base through natural-language queries, supported by citations.
