from setuptools import setup, find_packages
import os
import versioneer


def setup_package():
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()

    setup(
        name='neural-monitor',
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        description='Let me take care of your experiments statistics.',
        long_description=long_description,
        long_description_content_type='text/markdown',
        url='https://github.com/justanhduc/neural-monitor',
        author='Duc Nguyen',
        author_email='adnguyen@yonsei.ac.kr',
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: End Users/Desktop',
            'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
            'Operating System :: Microsoft :: Windows :: Windows 10',
            'Operating System :: POSIX :: Linux',
            'Programming Language :: Python :: 3 :: Only',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7'
        ],
        platforms=['Windows', 'Linux'],
        packages=find_packages(exclude=['docs']),
        install_requires=['matplotlib', 'numpy', 'imageio', 'tensorboard', 'git-python', 'easydict', 'pandas'],
        project_urls={
            'Bug Reports': 'https://github.com/justanhduc/neural-monitor/issues',
            'Source': 'https://github.com/justanhduc/neural-monitor',
        },
    )


if __name__ == '__main__':
    setup_package()
