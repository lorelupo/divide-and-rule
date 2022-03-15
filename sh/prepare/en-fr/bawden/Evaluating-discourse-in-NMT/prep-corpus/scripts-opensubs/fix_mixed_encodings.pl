#!/usr/bin/env perl
# -*- coding: utf-8 -*-

use strict;
use Encode;

binmode STDOUT , ":encoding(iso-8859-1)";

my $bool;

while (<>) {
  chomp;
  $_ = u8_to_8bits($_);

  s/\r//g;
  s/\l//g;

  $bool = 0;
  if (/&#194;/) {
	my $orig = $_;
	if (/^&#195;&#166;&#194;&#152;&#194;&#165;&#195;&#167;/
		||
		/^&#195;&#165;&#194;&#141;&#195;&#164;&#194;&#186;/)
	  {
		$_ = "-";
	  } else {
  
		s/&#195;&#130;&#194;&#129;&#195;&#131;&#194;&#128;/à/g;
		s/&#195;&#169;&#194;&#140;&#194;&#133;/é/g;
		s/&#195;&#168;&#194;&#132;&#194;&#191;/à/g;
		s/&#195;&#169;&#194;&#148;&#194;&#154;/ê/g;
		s/&#195;&#169;&#194;&#132;&#194;&#191;/à/g;
		s/&#194;&#129;&#195;&#128;/À/g;
		s/&#194;&#129;&#195;&#135;/Ç/g;
		s/&#195;&#167;&#194;&#140;"/è/g;
		s/&#194;&#129;f/’/g;
	
		s/&#194;&#129;&#195;&#180;/-/g; #???
		s/&#195;&#132;&#194;&#129;/ā/g;
	
		s/&#194;&#128;/ç/g;
		s/&#194;&#129;//g;		#!!
		s/&#194;&#130;/é/g;
		s/&#194;&#131;/â/g;
		s/&#194;&#133;/à/g;
		s/&#194;&#135;/ç/g;
		s/&#194;&#134;//g; # UNKNOWN, IN FACT...
		s/&#194;&#136; /à /g;	# même combinaison que 136=ê
		s/&#194;&#136;/ê/g;		# ou à... d'où la ligne précédente...
		s/&#194;&#137;/â/g;
		s/&#194;&#138;/è/g;
		s/&#194;&#139;/ï/g;
		s/&#194;&#140;/î/g;
		s/&#194;&#141;/ç/g;
		s/&#194;&#142;/é/g;
		s/&#194;&#143;/è/g;
		s/&#194;&#144;/é/g;
		s/&#194;&#145;/ë/g;
		s/&#195;&#146;(tes|tre)/ê$1/g;
		s/&#194;&#146;/’/g;
		s/´/'/g;
		s/&#194;&#147;/ô/g;
		s/&#194;&#148;\b/”/g;
		s/^&#194;&#148;/”/g;	#?
		s/&#194;&#148;/î/g;
		s/\b&#194;&#149;([^ ])/“$1/g;
		s/([aeiou])&#194;&#149;/$1ï/g;
		s/&#194;&#149;/ô/g;	# ou ï ou  “... d'où les lignes précédentes...
		s/&#194;&#150;/û/g;
		s/o&#194;&#151;/où/g;
		s/&#194;&#151;/-/g;		#?
		s/&#194;&#153;/ô/g;
		s/&#194;&#154;/š/g;
		s/&#194;&#156;/œ/g;
		s/&#194;&#157;/ù/g;
		s/&#194;&#158;/û/g;

		s/œ/oe/g; # Protection against recode
	  }
	if (/&#194;&#(\d+);/ && $1 < 160)  {
	  $bool = 1;
	  print STDERR "> $orig\n";
	  print STDERR "> $_\n";
	}
  }

  my $o = "";
  while ($_ ne "") {
	if (s/^&#(\d+);//ms) {
	  $o .= chr($1);
	} else {
	  s/^(.)//sm;
	  $o .= $1;
	}
  }
  $_ = $o;
  print STDERR "$_\n" if $bool;
  print "$_\n";
}

sub u8_to_8bits {
  my $s = shift;
  $_ = Encode::decode('iso-8859-1', $s);
  s/([\x{80}-\x{FFFF}])/'&#' . ord($1) . ';'/gse;
  return $_;
}