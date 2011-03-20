package ToyBox::MLPerceptron;

use strict;
use warnings;

our $VERSION = '0.0.2';

sub new {
    my $class = shift;
    my $self = {
        dnum => 0,
        fdata => [],
        ldata => [],
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
        my $copy_l = $l;
        $self->{lindex}{$l} = scalar(keys %{$self->{lindex}})
            unless defined($self->{lindex}{$l});
        push(@{$self->{fdata}}, \%copy_attr);
        push(@{$self->{ldata}}, $copy_l);
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

    foreach my $t (1 .. $T) {
        my $miss_num = 0;
        for (my $i = 0; $i < $self->{dnum}; $i++) {
            my $attributes = $self->{fdata}[$i];
            my $label = $self->{ldata}[$i];

            my $max_score = 0;
            my $max_feature = {};
            my $max_l = undef;
            my $true_feature = {};
            foreach my $l (keys %{$self->{lindex}}) {
                my $feature = {};
                my $score = 0;
                while (my ($f, $val) = each %$attributes) {
                    my $key = "$f\001$l";
                    $feature->{$key} = $val;
                    $score += $alpha->{$key} * $val if defined($alpha->{$key});
                }
                if ($score >= $max_score) {
                    $max_score = $score;
                    $max_feature = $feature;
                    $max_l = $l;
                }
                $true_feature = $feature if $l eq $label;
            }
            die "max_l is undef" unless defined($max_l);
            if ($max_l ne $label) {
                if ($avg) {
                    while (my ($f, $v) = each %$alpha) {
                        $alpha_sum->{$f} += $continue_num * $v;
                    }
                    $update_num += $continue_num;
                    $continue_num = 1;
                }

                while (my ($f, $val) = each %$true_feature) {
                    $alpha->{$f} += $val;
                }
                while (my ($f, $val) = each %$max_feature) {
                    $alpha->{$f} -= $val;
                }
                $miss_num++;
            } else {
                $continue_num++ if $avg;
            }
        }

        print STDERR "t: $t, miss num: $miss_num\n" if $verbose;
        last if $miss_num == 0;

    }

    if ($avg) {
        $update_num += $continue_num;
        foreach my $f (keys %$alpha_sum) {
            $alpha_sum->{$f} += $continue_num * $alpha->{$f};
            $alpha_sum->{$f} /= $update_num;
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

    my $score = {};
    foreach my $l (keys %{$self->{lindex}}) {
        $score->{$l} = 0;
        while (my ($f, $val) = each %$attributes) {
            my $key = "$f\001$l";
            $score->{$l} += $alpha->{$key} * $val
                if defined($alpha->{$key});
        }
    }
    $score;
}

sub labels {
    my $self = shift;
    keys %{$self->{lindex}};
}

1;
__END__


=head1 NAME

ToyBox::MLPerceptron - Classifier using Multi Label Perceptron

=head1 SYNOPSIS

  use ToyBox::MLPerceptron;

  my $pct= ToyBox::MLPerceptron->new();
  
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

