import os
from weasyprint import HTML

# Content based on user's specific data
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <style>
        @page {{
            size: 841mm 1189mm; /* A0 Portrait */
            margin: 0;
            background-color: #f4f7f9;
        }}
        body {{
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 40px;
            color: #333;
            display: block;
        }}
        .header {{
            background-color: #1a3a5f;
            color: white;
            padding: 60px;
            text-align: center;
            border-bottom: 15px solid #4caf50;
            margin-bottom: 40px;
        }}
        .header h1 {{
            font-size: 80pt;
            margin: 0;
            text-transform: uppercase;
            letter-spacing: 2px;
        }}
        .header h2 {{
            font-size: 35pt;
            font-weight: 300;
            margin-top: 20px;
            color: #d1d9e6;
        }}
        .authors {{
            font-size: 28pt;
            margin-top: 15px;
        }}
        .main-container {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 40px;
            height: 950mm;
        }}
        .column {{
            display: flex;
            flex-direction: column;
            gap: 30px;
        }}
        .section {{
            background: white;
            padding: 35px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .section h3 {{
            font-size: 32pt;
            color: #1a3a5f;
            border-left: 12px solid #4caf50;
            padding-left: 20px;
            margin-top: 0;
            margin-bottom: 25px;
            text-transform: uppercase;
        }}
        p, li {{
            font-size: 22pt;
            line-height: 1.6;
            margin-bottom: 20px;
        }}
        .highlight-box {{
            background-color: #e8f5e9;
            border: 2px dashed #4caf50;
            padding: 25px;
            font-style: italic;
            border-radius: 8px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 25px 0;
            font-size: 18pt;
        }}
        th {{
            background-color: #1a3a5f;
            color: white;
            padding: 15px;
            text-align: left;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .diagram-placeholder {{
            width: 100%;
            height: 350px;
            background-color: #eee;
            border: 3px solid #ccc;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24pt;
            color: #666;
            margin: 20px 0;
            text-align: center;
            border-radius: 10px;
        }}
        .math {{
            font-family: "Times New Roman", Times, serif;
            font-size: 26pt;
            text-align: center;
            padding: 20px;
            background: #f0f4f8;
            border-radius: 10px;
        }}
        .footer {{
            position: absolute;
            bottom: 30px;
            left: 40px;
            right: 40px;
            text-align: center;
            font-size: 18pt;
            color: #777;
            border-top: 2px solid #ddd;
            padding-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Precision-Elastic DNN Acceleration</h1>
        <h2>A Co-Design of Radix-4 Hardware and NSGA-II Optimization</h2>
        <div class="authors">Siddharth Joshi, Nishant, Nishit, Atharva <br> Department of Engineering</div>
    </div>

    <div class="main-container">
        <div class="column">
            <div class="section">
                <h3>1. Abstract & Motivation</h3>
                <p>Deep Neural Networks (DNNs) exhibit significant <b>error resilience</b> at the layer level. Standard fixed-precision hardware (FP16/INT8) leads to "over-provisioning," where LSB energy is wasted on computations that do not improve classification accuracy.</p>
                <div class="highlight-box">
                    <b>Core Contribution:</b> A hardware-software co-design loop that dynamically assigns precision to each DNN layer based on accuracy tolerance, utilizing a custom <b>Precision-Gated Radix-4 Multiplier</b>.
                </div>
                <p>We evaluate our framework on <b>CIFAR-10</b> using VGG11, ResNet18, and AlexNet architectures.</p>
            </div>

            <div class="section">
                <h3>2. Hardware Architecture</h3>
                <p>The <b>Precision-Gated Radix-4 Core</b> is designed for elasticity. Unlike static multipliers, it employs dynamic masking and Integrated Clock Gating (ICG) to kill switching activity in unused bits.</p>
                <div class="diagram-placeholder">
                    [DIAGRAM: Radix-4 Booth Multiplier with Precision-Masking Logic]
                </div>
                <ul>
                    <li><b>Radix-4 Booth Encoding:</b> Reduces partial products by 50%.</li>
                    <li><b>Dynamic Exponent Unit:</b> Runtime calculation of Bias and Max boundaries based on <i>exp_bits</i>.</li>
                    <li><b>Gated Datapath:</b> Prevents LSB toggling for low-precision modes.</li>
                </ul>
                <div class="math">
                    Booth PP &in; {{0, &plusmn;M, &plusmn;2M}}
                </div>
            </div>
        </div>

        <div class="column">
            <div class="section">
                <h3>3. The Co-Design Flow</h3>
                <p>We bridge the gap between gate-level power and neural network accuracy through a three-stage optimization pipeline:</p>
                <ol>
                    <li><b>Search (NSGA-II):</b> Evolutionary search for Pareto-optimal precision configs (Total Bits, Exp Bits).</li>
                    <li><b>RTL Synthesis (Cadence Genus):</b> Synthesis on <b>65nm technology</b> to extract exact Power, Area, and Latency for every precision point.</li>
                    <li><b>Mapping (Timeloop):</b> Cycle-accurate simulation of data movement and compute energy.</li>
                </ol>
            </div>

            <div class="section">
                <h3>4. Multiplier Characterization (65nm)</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Config</th>
                            <th>Total Power (&mu;W)</th>
                            <th>Latency (ns)</th>
                            <th>Energy (pJ)</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr><td>5b_3e</td><td>118.4</td><td>2.48</td><td>0.29</td></tr>
                        <tr><td>9b_4e</td><td>112.5</td><td>2.69</td><td>0.30</td></tr>
                        <tr><td>16b_8e</td><td>113.6</td><td>13.74</td><td>1.56</td></tr>
                    </tbody>
                </table>
                <p><i>Note: The 16-bit mode exhibits a 5&times; energy penalty compared to 5-bit precision, highlighting the savings potential in FC layers.</i></p>
                <div class="diagram-placeholder" style="height: 250px;">
                    [INSERT GRAPH: Power vs. Precision Curve]
                </div>
            </div>
        </div>

        <div class="column">
            <div class="section">
                <h3>5. System-Level Evaluation</h3>
                <p>Performance analysis of <b>AlexNet</b> on CIFAR-10 using the Timeloop infrastructure.</p>
                <table>
                    <thead>
                        <tr>
                            <th>Layer</th>
                            <th>Config</th>
                            <th>TOPS/W</th>
                            <th>Energy/Comp (pJ)</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr><td>Conv1</td><td>9b_4e</td><td>20.76</td><td>2.74</td></tr>
                        <tr><td>Conv4</td><td>6b_4e</td><td>38.21</td><td>2.54</td></tr>
                        <tr><td>FC8</td><td>5b_3e</td><td>0.13</td><td>3.77</td></tr>
                    </tbody>
                </table>
                <div class="diagram-placeholder" style="height: 400px;">
                    [INSERT IMAGE: Accuracy vs. Energy Pareto Front for all NNs]
                </div>
                <p>Early layers require higher precision (9b_4e) for feature extraction, while FC layers tolerate aggressive scaling (5b_3e).</p>
            </div>

            <div class="section">
                <h3>6. Conclusion</h3>
                <ul>
                    <li><b>Elasticity:</b> Reconfigurable hardware allows "Goldilocks" precision—not too much, not too little.</li>
                    <li><b>Efficiency:</b> Achieved massive energy reduction with &lt;1% accuracy drop.</li>
                    <li><b>Future Work:</b> Scaling to Transformer-based LLMs and real-time hardware-in-the-loop adaptation.</li>
                </ul>
            </div>
        </div>
    </div>

    <div class="footer">
        Presented at [Conference Name 2026] &bull; This research was supported by [Your University/Funding] &bull; Poster designed for Best Poster Award competition.
    </div>
</body>
</html>
"""

# Save to PDF
output_path = "research_poster_siddharth_joshi.pdf"
HTML(string=html_content).write_pdf(output_path)