package ::MLPerceptron;

use strict;
use warnings;

our $VERSION = '0.0.1';

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

    my $alpha = $self->{alpha};

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
                    my $key = "$f$l";
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
                while (my ($f, $val) = each %$true_feature) {
                    $alpha->{$f} += $val;
                }
                while (my ($f, $val) = each %$max_feature) {
                    $alpha->{$f} -= $val;
                }
                $miss_num++;
            }
        }
        print STDERR "t: $t, miss num: $miss_num\n" if $verbose;
        last if $miss_num == 0;
    }
    1;
}

sub predict {
    my ($self, %params) = @_;

    my $attributes = $params{attributes} or die "No params: attributes";
    die "attributes is not hash ref"   unless ref($attributes) eq 'HASH';
    die "attributes is empty hash ref" unless keys %$attributes;

    my $alpha = $self->{alpha};

    return {} unless %$alpha;

    my $score = {};
    foreach my $l (keys %{$self->{lindex}}) {
        while (my ($f, $val) = each %$attributes) {
            my $key = "$f$l";
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
  
  $pct->train(T => 10, progress_cb => 'verbose');
  
  my $score = $pct->predict(
                  attributes => {a => 1, b => 1, d => 1, e =>1}
              );

=head1 DESCRIPTION

=head1 AUTHOR

TAGAMI Yukihiro <tagami.yukihiro@gmail.com>

=head1 LICENSE

This library is distributed under the term of the MIT license.

L<http://opensource.org/licenses/mit-license.php>
