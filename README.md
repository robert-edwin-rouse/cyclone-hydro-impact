# cyclone-hydro-impact

<a name="readme-top"></a>


<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]


<!-- PROJECT HEADER -->
<p align="center">
<img src="./assets/Figure1.png" width="250" height="250">
</p>
<br />
<div align="center">
    <a href="https://github.com/robert-edwin-rouse/cyclone-hydro-impact/issues">Report Bug</a>
    ·
    <a href="https://github.com/robert-edwin-rouse/cyclone-hydro-impact/issues">Request Feature</a>
</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Prerequisites

The project requires Python 3.8+, and various libraries described in the
[requirements.txt](https://github.com/robert-edwin-rouse/cyclone-hydro-impact/blob/main/requirements.txt)
file.

### Installation and Running

1. Clone the repository:
```sh
git clone https://github.com/robert-edwin-rouse/cyclone-hydro-impact.git
cd cyclone-hydro-impact
```

2. Install the dependencies. We recommend doing this from within a virtual environment, e.g.
```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
(If you do not wish to use a virtual environment then just the last step can be used to install the dependencies globally).

3. Download data and add it to the data folder from the following links, amending the configuration file as necessary with your email address and the names of downloaded datasets:
* [NOAA's International Best Track Archive for Climate Stewardship (IBTrACS) data](https://doi.org/10.25921/82ty-9e16)

4. Run the script to compile the tracking database and pre-process all of the data:
```sh
python3 data_retrieval.py
python3 compiler.py
```

5. Running the following scripts will generate all of the results from the accompanying paper:
```sh
python3 train.py
python3 predict.py
python3 postprocessing.py
```

Once the model has been trained and saved, a new input array, created using compiler.py, can be passed to the loaded model with the normalisation data, if used during training, to create predictions for future events as per the example in userexample.py.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTRIBUTING -->
## Contributing

Contributions are welcome.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* This project received support from Schmidt Sciences, LLC and Inigo Ltd via the [Institute of Computing for Climate Science](https://iccs.cam.ac.uk/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/robert-edwin-rouse/cyclone-hydro-impact.svg?style=for-the-badge
[contributors-url]: https://github.com/robert-edwin-rouse/cyclone-hydro-impact/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/robert-edwin-rouse/cyclone-hydro-impact.svg?style=for-the-badge
[forks-url]: https://github.com/robert-edwin-rouse/cyclone-hydro-impact/network/members
[stars-shield]: https://img.shields.io/github/stars/robert-edwin-rouse/cyclone-hydro-impact.svg?style=for-the-badge
[stars-url]: https://github.com/robert-edwin-rouse/cyclone-hydro-impact/stargazers
[issues-shield]: https://img.shields.io/github/issues/robert-edwin-rouse/cyclone-hydro-impact.svg?style=for-the-badge
[issues-url]: https://github.com/robert-edwin-rouse/cyclone-hydro-impact/issues
[license-shield]: https://img.shields.io/github/license/robert-edwin-rouse/cyclone-hydro-impact.svg?style=for-the-badge
[license-url]: https://github.com/robert-edwin-rouse/cyclone-hydro-impact/LICENSE.txt
[product-screenshot]: ./assets/Figure1.png