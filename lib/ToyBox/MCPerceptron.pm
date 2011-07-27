package ToyBox::MCPerceptron;

use strict;
use warnings;

use List::Util qw(shuffle);

our $VERSION = '0.0.2';

sub new {
    my $class = shift;
    my $self = {
        dnum => 0,
        data => [],
        lindex => {},
        alpha => {},
    } ;
    bless $self, $class;
}

sub add_instance {
    my ($self, %params) = @_;

    my $attributes = $params{attributes} or die "No params: attributes";
    my $label      = $params{label}     or die "No params: label";
    die "attributes is not hash ref"   unless ref($attributes) eq 'HASH';
    die "attributes is empty hash ref" unless keys %$attributes;
    $label = [$label] unless ref($label) eq 'ARRAY';

    my %copy_attr = %$attributes;

    foreach my $l (@$label) {
        $self->{lindex}{$l} = scalar(keys %{$self->{lindex}})
            unless defined($self->{lindex}{$l});
        my $datum = {f => \%copy_attr, l => $l};
        push(@{$self->{data}}, $datum);
        $self->{dnum}++;
    }
}

sub train {
    my ($self, %params) = @_;

    my $T = $params{T};
    $T = 10 unless defined($T);
    die "T is le 0" unless scalar($T) > 0;

    my $progress_cb = $params{progress_cb};
    my $verbose = 0;
    $verbose = 1 if defined($progress_cb) && $progress_cb eq 'verbose';

    my $algorithm = $params{algorithm};
    my $avg = 0;
    $avg = 1 if defined($algorithm) && $algorithm eq 'average';

    my $alpha = $self->{alpha};
    my $alpha_sum = {};
    my $update_num = 0;

    my $continue_num = 1;

    my @all_labels = keys %{$self->{lindex}};

    foreach my $t (1 .. $T) {
        my $miss_num = 0;
        foreach my $datum (@{$self->{data}}) {
            my $attributes = $datum->{f};
            my $label = $datum->{l};

            my $scores = {};
            while (my ($f, $val) = each %$attributes) {
                next unless $alpha->{$f};
                while (my ($l, $w) = each %{$alpha->{$f}}) {
                    $scores->{$l} += $val * $w;
                }
            }

            my @sorted_label =
                sort {$scores->{$b} <=> $scores->{$a}} keys %$scores;

            my $max_l = $sorted_label[0] || $all_labels[0];

            if ($max_l ne $label) {
                if ($avg) {
                    foreach my $f (keys %$alpha) {
                        while (my ($l, $w) = each %{$alpha->{$f}}) {
                            $alpha_sum->{$f}{$l} += $continue_num * $w;
                        }
                    }
                    $update_num += $continue_num;
                    $continue_num = 1;
                }
                while (my ($f, $val) = each %$attributes) {
                    $alpha->{$f}{$label} += $val;
                    $alpha->{$f}{$max_l} -= $val;
                }
                $miss_num += 1;
            } else {
                $continue_num += 1 if $avg;
            }
        }
        print STDERR "t: $t, miss num: $miss_num\n" if $verbose;
        last if $miss_num == 0;

    }

    if ($avg) {
        $update_num += $continue_num;
        foreach my $f (keys %$alpha) {
            while (my ($l, $w) = each %{$alpha->{$f}}) {
                $alpha_sum->{$f}{$l} += $continue_num * $alpha->{$f}{$l};
                $alpha_sum->{$f}{$l} /= $update_num;
            }
        }
        $self->{alpha} = $alpha_sum;
    }

    1;
}

sub predict {
    my ($self, %params) = @_;

    my $attributes = $params{attributes} or die "No params: attributes";
    die "attributes is not hash ref"   unless ref($attributes) eq 'HASH';
    die "attributes is empty hash ref" unless keys %$attributes;

    my $alpha = $self->{alpha};

    my $scores = {};
    while (my ($f, $val) = each %$attributes) {
        next unless $alpha->{$f};
        while (my ($l, $w) = each %{$alpha->{$f}}) {
            $scores->{$l} += $val * $w;
        }
    }

    $scores;
}

sub labels {
    my $self = shift;
    keys %{$self->{lindex}};
}

1;
__END__


=head1 NAME

ToyBox::MCPerceptron - Classifier using MultiClass Perceptron Algorithm

=head1 SYNOPSIS

  use ToyBox::MCPerceptron;

  my $pct= ToyBox::MCPerceptron->new();
  
  $pct->add_instance(
      attributes => {a => 2, b => 3},
      label => 'positive'
  );
  
  $pct->add_instance(
      attributes => {c => 3, d => 1},
      label => 'negative'
  );
  
  $pct->train(T => 10,
              algorithm => 'average',
              progress_cb => 'verbose');
  
  my $score = $pct->predict(
                  attributes => {a => 1, b => 1, d => 1, e =>1}
              );

=head1 DESCRIPTION

=head1 AUTHOR

TAGAMI Yukihiro <tagami.yukihiro@gmail.com>

=head1 LICENSE

This library is distributed under the term of the MIT license.

L<http://opensource.org/licenses/mit-license.php>

