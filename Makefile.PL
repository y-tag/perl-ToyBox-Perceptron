use inc::Module::Install;

name          'ToyBox-Perceptron';
all_from      'lib/ToyBox/Perceptron.pm';
license       'MIT';

test_requires 'Test::More';

WriteAll;
