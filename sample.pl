#/usr/bin/perl

use strict;
use warnings;

use Data::Dumper;

use lib('lib');
use ToyBox::Perceptron;

my $pct = ToyBox::Perceptron->new();

my @pos_data = qw(a b ab bb aaab);
my @neg_data = qw(cc d cd ddd ccd);

foreach my $d (@pos_data) {
    my $tmp = make_attributes($d);
    $pct->add_instance(attributes => $tmp, label => 'positive');
}
foreach my $d (@neg_data) {
    my $tmp = make_attributes($d);
    $pct->add_instance(attributes => $tmp, label =>'negative');
}

$pct->train(T => 10, progress_cb => 'verbose');

foreach my $d qw(abccd adf) {
    my $tmp   = make_attributes($d);
    my $score = $pct->predict(attributes => $tmp);
    print Dumper($d, $score);
}

print Dumper($pct->labels);


sub make_attributes {
    my $data = shift;

    my $attributes = {};
    foreach my $chr (split(//, $data)) {
        $attributes->{$chr}++;
    }

    $attributes;
}

